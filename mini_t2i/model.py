from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    # RMSNorm：按最后一维的均方根做归一化，可选学习到的逐通道缩放参数。
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        # 初始化归一化稳定项和可学习权重；关闭 affine 时只做纯 RMS 归一化。
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将输入缩放到 RMS 约为 1，再按需乘以可学习权重。
        y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y if self.weight is None else y * self.weight


class SwiGLU(nn.Module):
    # SwiGLU 前馈层：用 SiLU 门控分支调制值分支，再投影回原 hidden 维度。
    def __init__(self, dim: int, hidden_dim: int):
        # hidden_dim 向上对齐到 8 的倍数，便于张量核心/向量化实现更友好。
        super().__init__()
        hidden_dim = math.ceil(hidden_dim / 8) * 8
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w3.weight)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # w1 产生门控信号，w3 产生值分支，两者逐元素相乘后由 w2 回到输入维度。
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: int = 10000) -> torch.Tensor:
    # 将连续扩散/flow 时间 t 编码为 sin/cos 频率特征，供时间 MLP 使用。
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    # 时间嵌入器：先做固定正弦时间编码，再用两层 MLP 映射到模型 hidden_size。
    def __init__(self, hidden_size: int):
        # 初始化时间 MLP；输出会和文本 pooled 条件一起作为全局时间/条件向量。
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # 生成 256 维正弦时间编码，并转换到 MLP 参数 dtype 后得到 hidden_size 向量。
        return self.mlp(timestep_embedding(t, 256).to(dtype=next(self.parameters()).dtype))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # RoPE 辅助函数：将最后一维拆成两半并旋转，用于构造复数旋转的等价实数形式。
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_1d_rope(x: torch.Tensor, start: int = 0, theta: int = 10000) -> torch.Tensor:
    # 对形状 [B, H, N, D] 的序列注意力张量施加一维 Rotary Position Embedding。
    b, h, n, d = x.shape
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    pos = torch.arange(start, start + n, device=x.device, dtype=torch.float32)
    angles = torch.einsum("n,f->nf", pos, inv)
    angles = torch.cat([angles, angles], dim=-1)
    cos = angles.cos()[None, None].to(x.dtype)
    sin = angles.sin()[None, None].to(x.dtype)
    return x * cos + rotate_half(x) * sin


def apply_2d_rope_flat(x: torch.Tensor, grid: int, theta: int = 10000) -> torch.Tensor:
    # 对展平的图像 token 施加二维 RoPE；若 token 数不是 grid^2，则退回一维 RoPE。
    b, h, n, d = x.shape
    if n != grid * grid:
        return apply_1d_rope(x, theta=theta)
    rope_dim = d // 2
    inv = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, device=x.device, dtype=torch.float32) / rope_dim))
    t = torch.arange(grid, device=x.device, dtype=torch.float32)
    freqs = torch.einsum("n,f->nf", t, inv)
    f_h, f_w = torch.broadcast_tensors(freqs[:, None, :], freqs[None, :, :])
    angles = torch.cat([f_h, f_w], dim=-1)
    angles = torch.cat([angles, angles], dim=-1).reshape(n, d)
    cos = angles.cos()[None, None].to(x.dtype)
    sin = angles.sin()[None, None].to(x.dtype)
    return x * cos + rotate_half(x) * sin


def apply_2d_rope_flat_compat(x: torch.Tensor, grid: int, theta: int = 10000) -> torch.Tensor:
    # 兼容式二维 RoPE：分别给高度和宽度半维特征加旋转位置编码。
    b, h, n, d = x.shape
    if n != grid * grid:
        return apply_1d_rope(x, theta=theta)
    half = d // 2
    x_h, x_w = x[..., :half], x[..., half:]
    yy = torch.arange(grid, device=x.device, dtype=torch.float32).repeat_interleave(grid)
    xx = torch.arange(grid, device=x.device, dtype=torch.float32).repeat(grid)
    inv = 1.0 / (theta ** (torch.arange(0, half, 2, device=x.device, dtype=torch.float32) / half))
    ah = torch.einsum("n,f->nf", yy, inv)
    aw = torch.einsum("n,f->nf", xx, inv)
    ah = torch.cat([ah, ah], dim=-1)
    aw = torch.cat([aw, aw], dim=-1)
    x_h = x_h * ah.cos()[None, None].to(x.dtype) + rotate_half(x_h) * ah.sin()[None, None].to(x.dtype)
    x_w = x_w * aw.cos()[None, None].to(x.dtype) + rotate_half(x_w) * aw.sin()[None, None].to(x.dtype)
    return torch.cat([x_h, x_w], dim=-1)


def attention_einsum(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head_dim: int) -> torch.Tensor:
    # 手写 einsum 注意力实现：计算缩放点积、softmax 权重，再聚合 value。
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * (head_dim**-0.5)
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", weights, v)


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_dim: int,
    impl: str,
) -> torch.Tensor:
    # 根据配置在 PyTorch SDPA 和显式 einsum 注意力之间分发，便于兼顾速度和兼容性。
    if impl == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=head_dim**-0.5)
    if impl == "einsum":
        return attention_einsum(q, k, v, head_dim)
    raise ValueError(f"unknown attention_impl={impl!r}")


class BottleneckPatchEmbed(nn.Module):
    # 图像 patch 嵌入：先用大步长卷积提取 patch，再用 1x1 卷积投影到 hidden_size。
    def __init__(self, image_size: int, patch_size: int, in_channels: int, hidden_size: int, pca_channels: int):
        # grid 是图像被 patch_size 划分后的边长；输出 token 数为 grid * grid。
        super().__init__()
        self.grid = image_size // patch_size
        self.proj1 = nn.Conv2d(in_channels, pca_channels, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_channels, hidden_size, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.xavier_uniform_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 [B, C, H, W]，输出展平后的图像 token [B, grid*grid, hidden_size]。
        x = self.proj2(self.proj1(x))
        return x.flatten(2).transpose(1, 2)


def sincos_2d(embed_dim: int, grid: int) -> torch.Tensor:
    # 生成固定二维 sin/cos 位置编码，给图像 patch token 提供初始空间位置信息。
    y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    omega = torch.arange(embed_dim // 4, dtype=torch.float32) / (embed_dim // 4)
    omega = 1.0 / (10000 ** omega)
    out_y = torch.einsum("n,d->nd", y.flatten().float(), omega)
    out_x = torch.einsum("n,d->nd", x.flatten().float(), omega)
    return torch.cat([out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=1)


class PlainTextBlock(nn.Module):
    # 纯文本 Transformer block：在进入图文联合注意力前，先独立处理文本 token。
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float,
        qk_norm: bool = True,
        rms_affine: bool = True,
        attention_impl: str = "einsum",
    ):
        # 构造文本自注意力、RMSNorm 和 SwiGLU MLP；可选对 Q/K 做 RMS 归一化。
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qk_norm = qk_norm
        self.attention_impl = attention_impl
        inner = num_heads * head_dim
        self.norm1 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.norm2 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.qkv = nn.Linear(hidden_size, inner * 3)
        self.proj = nn.Linear(inner, hidden_size)
        self.mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, elementwise_affine=rms_affine)
            self.k_norm = RMSNorm(head_dim, elementwise_affine=rms_affine)

    def forward(self, txt: torch.Tensor) -> torch.Tensor:
        # 文本 token 做自注意力和 MLP 残差更新，形状保持 [B, text_len, hidden_size]。
        b, n, _ = txt.shape
        qkv = self.qkv(self.norm1(txt)).view(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k, v = [z.transpose(1, 2) for z in (q, k, v)]
        q = apply_1d_rope(q)
        k = apply_1d_rope(k)
        out = attention_forward(q, k, v, self.head_dim, self.attention_impl)
        txt = txt + self.proj(out.transpose(1, 2).reshape(b, n, -1))
        return txt + self.mlp(self.norm2(txt))


class DoubleStreamBlock(nn.Module):
    # 图文双流 block：文本 token 与图像 token 拼接后做联合注意力，再分别回写两条流。
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float,
        image_grid: int,
        qk_norm: bool = True,
        rms_affine: bool = True,
        rope_style: str = "jax",
        attention_impl: str = "einsum",
    ):
        # 分别为图像流和文本流准备 QKV、输出投影、MLP；注意力中二者共享同一上下文。
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.image_grid = image_grid
        self.qk_norm = qk_norm
        self.rope_style = rope_style
        self.attention_impl = attention_impl
        inner = num_heads * head_dim
        self.img_norm1 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.img_norm2 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.txt_norm1 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.txt_norm2 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.img_qkv = nn.Linear(hidden_size, inner * 3)
        self.txt_qkv = nn.Linear(hidden_size, inner * 3)
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, elementwise_affine=rms_affine)
            self.k_norm = RMSNorm(head_dim, elementwise_affine=rms_affine)
        self.img_proj = nn.Linear(inner, hidden_size)
        self.txt_proj = nn.Linear(inner, hidden_size)
        self.img_mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        self.txt_mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))

    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 输入图像 token 和文本 token；联合注意力后返回更新后的两路 token。
        b, li, _ = img.shape
        lt = txt.shape[1]
        qi, ki, vi = self.img_qkv(self.img_norm1(img)).view(b, li, 3, self.num_heads, self.head_dim).unbind(2)
        qt, kt, vt = self.txt_qkv(self.txt_norm1(txt)).view(b, lt, 3, self.num_heads, self.head_dim).unbind(2)
        q = torch.cat([qt, qi], dim=1).transpose(1, 2)
        k = torch.cat([kt, ki], dim=1).transpose(1, 2)
        v = torch.cat([vt, vi], dim=1).transpose(1, 2)
        if self.qk_norm:
            q_text, q_img = self.q_norm(q[:, :, :lt]), self.q_norm(q[:, :, lt:])
            k_text, k_img = self.k_norm(k[:, :, :lt]), self.k_norm(k[:, :, lt:])
        else:
            q_text, q_img = q[:, :, :lt], q[:, :, lt:]
            k_text, k_img = k[:, :, :lt], k[:, :, lt:]
        rope2d = apply_2d_rope_flat_compat if self.rope_style == "compat" else apply_2d_rope_flat
        q = torch.cat([apply_1d_rope(q_text), rope2d(q_img, self.image_grid)], dim=2)
        k = torch.cat([apply_1d_rope(k_text), rope2d(k_img, self.image_grid)], dim=2)
        out = attention_forward(q, k, v, self.head_dim, self.attention_impl)
        out = out.transpose(1, 2)
        txt = txt + self.txt_proj(out[:, :lt].reshape(b, lt, -1))
        img = img + self.img_proj(out[:, lt:].reshape(b, li, -1))
        img = img + self.img_mlp(self.img_norm2(img))
        txt = txt + self.txt_mlp(self.txt_norm2(txt))
        return img, txt


class MMJiTB32Text2(nn.Module):
    # MiniT2I 的 MM-JiT 主干：直接在 RGB 像素空间预测图像，条件来自 T5 文本特征和时间 t。
    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 32,
        in_channels: int = 3,
        hidden_size: int = 768,
        t5_hidden_size: int = 1024,
        depth_double: int = 17,
        text_preamble_depth: int = 2,
        num_heads: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 2.6667,
        pca_channels: int = 128,
        final_layer_zero: bool = True,
        rms_affine: bool = True,
        text_qk_norm: bool = True,
        double_qk_norm: bool = True,
        rope_style: str = "jax",
        attention_impl: str = "einsum",
    ):
        # 搭建图像 patch 嵌入、文本投影、时间嵌入、文本预处理 block 和图文联合 block。
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid = image_size // patch_size
        self.img_embed = BottleneckPatchEmbed(image_size, patch_size, in_channels, hidden_size, pca_channels)
        self.txt_embed = nn.Linear(t5_hidden_size, hidden_size, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, t5_hidden_size))
        nn.init.normal_(self.mask_token, std=0.02)
        self.t_embed = TimestepEmbedder(hidden_size)
        self.pooled_embed = nn.Linear(t5_hidden_size, hidden_size, bias=False)
        self.register_buffer("pos_embed", sincos_2d(hidden_size, self.grid).unsqueeze(0), persistent=False)
        self.txt_blocks = nn.ModuleList(
            [
                PlainTextBlock(
                    hidden_size,
                    num_heads,
                    head_dim,
                    mlp_ratio,
                    qk_norm=text_qk_norm,
                    rms_affine=rms_affine,
                    attention_impl=attention_impl,
                )
                for _ in range(text_preamble_depth)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    hidden_size,
                    num_heads,
                    head_dim,
                    mlp_ratio,
                    self.grid,
                    qk_norm=double_qk_norm,
                    rms_affine=rms_affine,
                    rope_style=rope_style,
                    attention_impl=attention_impl,
                )
                for _ in range(depth_double)
            ]
        )
        self.final_norm = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.final = nn.Linear(hidden_size, patch_size * patch_size * in_channels)
        if final_layer_zero:
            nn.init.zeros_(self.final.weight)
            nn.init.zeros_(self.final.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        # 将每个 patch token 预测出的像素块重新拼回 [B, C, H, W] 图像张量。
        b, n, d = x.shape
        p = self.patch_size
        c = self.in_channels
        g = int(n**0.5)
        x = x.view(b, g, g, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, c, g * p, g * p)

    def forward(self, img: torch.Tensor, t: torch.Tensor, context: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # 主前向：输入带噪图像、时间 t、T5 文本 token 和 attention mask，输出预测图像。
        mask = attn_mask.to(dtype=torch.bool)[:, :, None]
        # mask 为 False 的文本位置替换成可学习 mask_token，避免 padding 直接进入模型。
        context = torch.where(mask, context, self.mask_token.to(dtype=context.dtype))
        # 图像转 patch token 后加固定二维位置编码。
        img_tokens = self.img_embed(img) + self.pos_embed.to(device=img.device, dtype=img.dtype)
        # 文本 token 投影到模型 hidden_size；pooled 文本和时间嵌入形成全局条件向量。
        txt = self.txt_embed(context)
        pooled = context.mean(dim=1)
        vec = self.t_embed(t).to(dtype=img.dtype) + self.pooled_embed(pooled)
        del vec
        # 先用纯文本 block 处理文本上下文，再进入图文双流联合建模。
        for block in self.txt_blocks:
            txt = block(txt)
        for block in self.blocks:
            img_tokens, txt = block(img_tokens, txt)
        # 只取图像 token 做最终像素块预测，再 unpatchify 回 RGB 图像。
        out = self.final(self.final_norm(img_tokens))
        return self.unpatchify(out).float()
