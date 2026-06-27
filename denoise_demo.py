#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "diffusers"))
sys.path.insert(0, str(ROOT / "lora"))


TARGET_IMAGE_SIZE = 512
TARGET_PATCH_SIZE = 16
NOISE_SCALE = 2.0
# 本 demo 固定演示 MiniT2I-B/16；即使其他模型也可能是 512/16，也不在这里混用。
B16_ALIASES = {
    "b",
    "b16",
    "b-16",
    "base",
    "minit2i-b16",
    "minit2i-b-16",
    "minit2i-b/16",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Add MiniT2I flow-matching noise to one image, then denoise it.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--prompt", default="", help="Text condition used during denoising.")
    parser.add_argument("--outdir", default="denoise_outputs", help="Directory for demo outputs.")
    parser.add_argument("--pretrained_model_name_or_path", default="MiniT2I/MiniT2I")
    parser.add_argument("--model_type", default="b16", help="MiniT2I model type. This demo expects B/16.")
    parser.add_argument("--steps", type=int, default=100, help="Euler denoising steps from t_start to 1.")
    parser.add_argument("--noise_strength", type=float, default=0.5, help="Noise strength in [0, 1].")
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--cache_dir", default=None)
    return parser.parse_args()


def load_runtime_deps():
    # 延迟导入深度学习依赖，让 `--help` 和基础参数错误不需要完整运行环境。
    global torch, Image, AutoTokenizer, T5EncoderModel
    global MiniT2IFlowMatchScheduler, find_minit2i_transformer, load_hf_transformer

    import torch
    from PIL import Image
    from transformers import AutoTokenizer, T5EncoderModel
    from transformers import logging as transformers_logging

    from pipeline import MiniT2IFlowMatchScheduler
    from train_lora import find_minit2i_transformer, load_hf_transformer

    transformers_logging.set_verbosity_error()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    return torch.bfloat16


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def load_image_tensor(path: str | os.PathLike[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    # MiniT2I-B/16 默认输入分辨率是 512x512；这里先中心裁成正方形再缩放。
    image = center_crop_square(image).resize((TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), Image.BICUBIC)
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    data = data.reshape(TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE, 3).permute(2, 0, 1)
    data = data.to(device=device, dtype=dtype).unsqueeze(0)
    # PIL 像素范围 [0, 255] 转成 MiniT2I 使用的 [-1, 1]。
    return data / 127.5 - 1.0


def tensor_to_pil(images: torch.Tensor) -> list[Image.Image]:
    images = (images.detach().float().clamp(-1, 1) * 127.5 + 128.0).clamp(0, 255).to(torch.uint8)
    arrays = images.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(array) for array in arrays]


def encode_prompt(tokenizer, text_encoder, prompt: str, cfg, device: torch.device, dtype: torch.dtype):
    tokens = tokenizer(
        [prompt],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=cfg.prompt_length,
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    with torch.no_grad():
        text = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    return text.to(dtype=dtype), attention_mask.to(device)


def validate_b16_config(cfg) -> None:
    # 明确检查实际加载到的权重配置，避免把其他分辨率/patch 的模型用于这个 demo。
    if cfg.image_size != TARGET_IMAGE_SIZE or cfg.patch_size != TARGET_PATCH_SIZE:
        raise ValueError(
            "denoise_demo.py is fixed to MiniT2I-B/16 defaults: "
            f"image_size={TARGET_IMAGE_SIZE}, patch_size={TARGET_PATCH_SIZE}. "
            f"Loaded config has image_size={cfg.image_size}, patch_size={cfg.patch_size}."
        )


def validate_model_type(model_type: str) -> None:
    key = model_type.lower().replace("_", "-")
    if key not in B16_ALIASES:
        choices = ", ".join(sorted(B16_ALIASES))
        raise ValueError(f"denoise_demo.py is fixed to MiniT2I-B/16. Use one of: {choices}.")


def inference_timesteps(
    scheduler: MiniT2IFlowMatchScheduler,
    t_start: float,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # MiniT2IFlowMatchScheduler 只提供 flow-matching 时间轴；真正的 Euler 更新在下方循环里做。
    if hasattr(scheduler, "get_inference_timesteps"):
        base = scheduler.get_inference_timesteps(steps, device=device, dtype=dtype)
    else:
        base = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)
    return t_start + (1.0 - t_start) * base


def denoise_from_t(
    model,
    x_t: torch.Tensor,
    t_start: float,
    text: torch.Tensor,
    attention_mask: torch.Tensor,
    scheduler: MiniT2IFlowMatchScheduler,
    steps: int,
    cfg_scale: float,
    progress: bool = True,
) -> torch.Tensor:
    # 从给定图片加噪后的中间时刻 t_start 开始，而不是像 model.sample() 那样从纯噪声 t=0 开始。
    dtype = x_t.dtype
    timesteps = inference_timesteps(scheduler, t_start, steps, x_t.device, dtype)
    iterator = range(steps)
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="Denoising")
    x = x_t
    for i in iterator:
        t_cur = timesteps[i].expand(x.shape[0])
        t_next = timesteps[i + 1].expand(x.shape[0])
        # 模型网络直接预测的是 x0；cfg_velocity 内部会先得到 x0，再换算成 Euler 需要的 velocity。
        v = model.model.cfg_velocity(
            x,
            t_cur,
            text.to(dtype),
            attention_mask.to(dtype),
            cfg_scale,
        )
        # Euler 积分：x_{t_next} = x_t + (t_next - t_cur) * v。
        x = x + (t_next - t_cur)[:, None, None, None] * v
    return x


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    if not 0.0 <= args.noise_strength <= 1.0:
        raise ValueError("--noise_strength must be in [0, 1].")
    validate_model_type(args.model_type)

    load_runtime_deps()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    transformer, cfg = load_hf_transformer(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        model_type=args.model_type,
        cache_dir=args.cache_dir,
        revision=None,
        variant=None,
        torch_dtype=dtype,
    )
    validate_b16_config(cfg)
    transformer.to(device=device, dtype=dtype).eval().requires_grad_(False)
    model = find_minit2i_transformer(transformer)

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm, cache_dir=args.cache_dir)
    text_encoder = T5EncoderModel.from_pretrained(cfg.llm, torch_dtype=torch.float32, cache_dir=args.cache_dir)
    text_encoder.to(device).eval().requires_grad_(False)

    image = load_image_tensor(args.image, device, dtype)
    text, attention_mask = encode_prompt(tokenizer, text_encoder, args.prompt, cfg, device, dtype)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(image.shape, device=device, dtype=dtype, generator=generator) * NOISE_SCALE
    # MiniT2I 的 flow-matching 约定：t=1 接近原图，t=0 接近纯噪声。
    t_start = 1.0 - args.noise_strength
    # 加噪公式和训练保持一致：x_t = t * x0 + (1 - t) * noise。
    x_t = image * t_start + noise * (1.0 - t_start)

    # scheduler 记录 MiniT2I 的时间步设置；下面的 denoise_from_t 才执行 Euler 去噪。
    scheduler = MiniT2IFlowMatchScheduler(num_inference_steps=args.steps)
    with torch.no_grad():
        denoised = denoise_from_t(
            model=model,
            x_t=x_t,
            t_start=t_start,
            text=text,
            attention_mask=attention_mask,
            scheduler=scheduler,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
        )

    tag = f"t{t_start:.3f}_noise{args.noise_strength:.3f}".replace(".", "p")
    tensor_to_pil(image)[0].save(outdir / "input.png")
    tensor_to_pil(x_t)[0].save(outdir / f"noisy_{tag}.png")
    tensor_to_pil(denoised)[0].save(outdir / f"denoised_{tag}.png")

    metadata = {
        "image": str(args.image),
        "prompt": args.prompt,
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "model_type": args.model_type,
        "image_size": TARGET_IMAGE_SIZE,
        "patch_size": TARGET_PATCH_SIZE,
        "steps": args.steps,
        "noise_strength": args.noise_strength,
        "t_start": t_start,
        "noise_scale": NOISE_SCALE,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "scheduler": {
            "name": "MiniT2IFlowMatchScheduler",
            "train_t_schedule": scheduler.config.train_t_schedule,
            "t_lognorm_mu": scheduler.config.t_lognorm_mu,
            "t_lognorm_sigma": scheduler.config.t_lognorm_sigma,
            "num_inference_steps": scheduler.config.num_inference_steps,
            "noise_formula": "x_t = t * x0 + (1 - t) * noise",
            "inference": "Euler integration from t_start to 1.0 with linear timesteps",
        },
        "outputs": {
            "input": "input.png",
            "noisy": f"noisy_{tag}.png",
            "denoised": f"denoised_{tag}.png",
        },
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved denoise demo outputs to {outdir}")


if __name__ == "__main__":
    main()
