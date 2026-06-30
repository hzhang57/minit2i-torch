<p align="center">
  <img src="assets/teaser.png" alt="MiniT2I logo" />
</p>

<h2 align="center">MiniT2I：一个最简文本到图像生成基线</h1>

<p align="center">
  <a href="https://peppaking8.github.io/#/post/minit2i"><img src="https://img.shields.io/badge/Blog-MiniT2I-2ea44f.svg" alt="MiniT2I blog post" /></a>
  &nbsp;
  <a href="https://huggingface.co/MiniT2I/MiniT2I"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoints-MiniT2I-yellow.svg" alt="Hugging Face checkpoints" /></a>
  &nbsp;
  <a href="https://github.com/PeppaKing8/minit2i-jax"><img src="https://img.shields.io/badge/Code-JAX-blue.svg" alt="JAX code" /></a>
  &nbsp;
  <a href="https://colab.research.google.com/github/Hope7Happiness/minit2i-torch/blob/main/notebooks/minit2i_colab_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" /></a>
  &nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License" /></a>
</p>

MiniT2I 的 PyTorch/Diffusers 复现实现。

MiniT2I 是一个简单的像素空间文本到图像生成器，它使用流匹配训练一个直接输出 RGB 的 **MM-JiT** 去噪器，并且以冻结的 FLAN-T5-Large 文本编码器作为条件。该方案刻意保持非常简洁：不使用图像 tokenizer、不采用级联生成、没有 RL 阶段，也没有任何辅助损失。MiniT2I 的训练数据完全来自公开来源，容易复现实现。更多细节请参考我们的 [博客文章](https://peppaking8.github.io/#/post/minit2i)。

该仓库包含 MiniT2I 的 PyTorch 复现代码：
- PyTorch/Diffusers 版本 MiniT2I-B/16 和 MiniT2I-L/16 的 **推理**。
- 下游适应实验中使用的 **LoRA** 适配代码。
- MiniT2I-B/32 消融设置的完整 PyTorch **训练**代码。
- GenEval、DPG-Bench 和 FID **评估**入口。

## 目录

- [模型库](#模型库)
- [仓库结构](#仓库结构)
- [安装](#安装)
- [Diffusers 推理 & Colab 演示](#diffusers-推理--colab-演示)
  - [推荐推理设置](#推荐推理设置)
- [LoRA 适配](#lora-适配)
- [评估](#评估)
  - [评估指标](#评估指标)
  - [评估检查点](#评估检查点)
- [完整训练](#完整训练)
  - [数据集准备](#数据集准备)
  - [CC12M 预训练](#cc12m-预训练)
  - [120K mix 微调](#120k-mix-微调)
- [致谢](#致谢)
- [引用](#引用)

## 模型库

| 模型 | 参数量 | Patch | GenEval | DPG-Bench |
| --- | ---: | ---: | ---: | ---: |
| MiniT2I-B/16 | 258M + 341M 文本编码器 | 16 | 0.873 (0.873) | 84.1 (84.2) |
| MiniT2I-L/16 | 912M + 341M 文本编码器 | 16 | 0.878 (0.883) | 84.9 (85.9) |

括号内为 [JAX 实现](https://github.com/PeppaKing8/minit2i-jax) 的复现结果。

模型权重可参见 Hugging Face 仓库：<https://huggingface.co/MiniT2I/MiniT2I>。该仓库存储了 MiniT2I-B/16 和 MiniT2I-L/16 的权重。

## 仓库结构

```text
.
├── main.py                     # 统一的训练/评估入口
├── configs/                    # 预训练、微调和评估配置
├── mini_t2i/                    # PyTorch 训练和基准评估代码
├── diffusers/                  # 自定义 Diffusers pipeline 源码
├── lora/                       # LoRA 适配入口
└── tools/                      # 数据集准备辅助脚本
```

## 安装

创建一个带 CUDA 支持的 Python 环境，然后安装依赖：

```bash
git clone <REPO_URL>
cd minit2i-torch
python -m pip install -r requirements.txt
```

如果仅用于推理，核心依赖为 `torch`、`diffusers`、`transformers`、`safetensors`、`pillow` 和 `huggingface_hub`。

在训练或评估前，编辑 `mini_t2i/settings.py` 顶部的用户设置：

```python
DATA_ROOT = Path("/path/to/dataset/root")
OUTPUT_ROOT = Path("/path/to/output/root")
CHECKPOINT_ROOT = Path("/path/to/checkpoint/root")
EVAL_OUTPUT_ROOT = Path("/path/to/evaluation/output/root")
ASSET_ROOT = Path("/path/to/evaluation/assets")

HF_TOKEN_FILE = Path("/path/to/secrets/hf_token")
WANDB_API_KEY_FILE = Path("/path/to/secrets/wandb_api_key")
```

`mini_t2i/settings.py` 的其余部分会根据这些根目录推导基准、数据集、检查点和输出路径。

## Diffusers 推理 & Colab 演示

可以直接通过 [Colab 演示](https://colab.research.google.com/github/Hope7Happiness/minit2i-torch/blob/main/notebooks/minit2i_colab_demo.ipynb) 体验 MiniT2I，而无需准备训练数据。

```python
import torch
from diffusers import DiffusionPipeline

HUB_MODEL_ID = "MiniT2I/MiniT2I"

pipe = DiffusionPipeline.from_pretrained(
    HUB_MODEL_ID,
    custom_pipeline=HUB_MODEL_ID,
    trust_remote_code=True,
)

image = pipe(
    "A lonely astronaut standing on a quiet beach under two moons.",
    model_type="b16",
    guidance_scale=2.5,
    num_inference_steps=100,
    torch_dtype=torch.bfloat16,
).images[0]
image.save("minit2i-b16.png")

image = pipe(
    "a transparent green backpack on a marble pedestal, with notebooks and a metal water bottle visible inside",
    model_type="l16",
    guidance_scale=6.0,
    num_inference_steps=100,
    torch_dtype=torch.bfloat16,
).images[0]
image.save("minit2i-l16.png")
```

支持的别名包括：
- `b`, `b16`, `base`, `minit2i-b/16`（表示 MiniT2I-B/16）；
- `l`, `l16`, `large`, `minit2i-l/16`（表示 MiniT2I-L/16）。

### 推荐推理设置

以下设置基于 Colab 演示和 GenEval / DPG-Bench 评估配置，适合作为起点；CFG 可以根据提示调整。

| 设置 | MiniT2I-B/16 | MiniT2I-L/16 | 来源 |
| --- | --- | --- | --- |
| `num_inference_steps` | 100 | 100 | Colab 演示 & 训练 `n_T` |
| `guidance_scale`（通用） | 2.5 | 6.0 | Colab 演示 |
| `guidance_scale`（GenEval / DPG-Bench） | 5.0 | 5.0 | `configs/eval_hf_*.yml` |
| `torch_dtype` | `bfloat16` | `bfloat16` | Colab 演示 |
| 分辨率 | 512 × 512 | 512 × 512 | 训练图像尺寸 |

## LoRA 适配

LoRA 代码会将适配器附加到注意力投影、MLP 投影和文本投影层。

这里给出博客中默认的设置示例：在 Naruto BLIP-caption 数据集上使用 rank-32 适配器作为域适应示例。

```bash
python lora/train_lora.py \
  --pretrained_model_name_or_path MiniT2I/MiniT2I \
  --model_type b16 \
  --dataset_name lambdalabs/naruto-blip-captions \
  --output_dir lora_runs/naruto_lora \
  --train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --rank 32 \
  --lora_alpha 32 \
  --mixed_precision bf16 \
  --loss_type velocity \
  --max_train_steps 400 \
  --validation_at_start \
  --validation_steps_interval 100 \
  --validation_prompt "anime portrait of a ninja with orange hair, high quality"
```

采样训练好的适配器：

```bash
python lora/sample_lora.py \
  --adapter_dir lora_runs/naruto_lora \
  --prompt "anime portrait of a ninja with orange hair, high quality" \
  --steps 100 \
  --cfg_scale 6.0 \
  --device cuda \
  --dtype bfloat16
```

如需端到端演示，请查看 `notebooks/minit2i_lora_naruto_finetune.ipynb`。

## 评估

评估管道支持 Hugging Face Diffusers 检查点和本地 PyTorch 检查点的 GenEval 与 DPG-Bench 提示跟随评估。

### 评估指标

本仓库未附带基准提示、检测器检查点或 VQA 检查点。请单独下载每个评价资源，并编辑 `mini_t2i/settings.py`。

**GenEval。** GenEval 使用来自 [`djghosh13/geneval`](https://github.com/djghosh13/geneval) 的提示元数据和检测器检查点。请参考该仓库下载提示元数据和预训练检测器模型。默认设置下，请将文件放置为：

```text
/path/to/evaluation/assets/geneval/evaluation_metadata.jsonl
/path/to/evaluation/assets/geneval/pretrained/
```

**DPG-Bench。** DPG-Bench 使用 Hugging Face 数据集 [`Jialuo21/DPG-Bench`](https://huggingface.co/datasets/Jialuo21/DPG-Bench) 以及 ModelScope 的 mPLUG VQA 检查点 [`iic/mplug_visual-question-answering_coco_large_en`](https://www.modelscope.cn/models/iic/mplug_visual-question-answering_coco_large_en/summary)。Hugging Face 数据集默认以 parquet 格式存储并包含 `test` 划分，因此默认配置使用 `datasets` 直接加载，而无需单独 CSV 下载。

默认设置下，请将 mPLUG 检查点放置为：

```text
/path/to/evaluation/assets/modelscope/iic/mplug_visual-question-answering_coco_large_en/
```

如需使用本地 DPG-Bench 文件，请在 `mini_t2i/settings.py` 中将 `DPG_BENCH_DATA` 设置为本地 CSV、parquet、JSON 或 JSONL 路径。

### 评估检查点

直接从 Hub 评估 Hugging Face Diffusers 检查点，无需下载本地 `.pt` 文件：

```bash
python main.py eval --config-file configs/eval_hf_b16.yml
python main.py eval --config-file configs/eval_hf_l16.yml
```

这些配置使用 `hf_model_id: MiniT2I/MiniT2I` 和 `model_type: b16` 或 `model_type: l16`。仅在评估本地 PyTorch 检查点时，编辑 `configs/eval.yml`：

```yaml
checkpoint: /path/to/checkpoint.pt
checkpoint_target: ema
output_dir: /path/to/eval_output
```

运行所有启用的基准：

```bash
python main.py eval --config-file configs/eval.yml
```

## 完整训练

PyTorch 训练接口通过 `main.py`、`configs/` 和 `mini_t2i/` 配置驱动。

本仓库提供 `configs/pretrain.yml` 和 `configs/finetune.yml` 中的默认 B/32 消融设置。B/16 和 L/16 的训练配置请参考我们的 [JAX 代码发布](https://github.com/PeppaKing8/minit2i-jax)。

### 数据集准备

本仓库不包含训练数据。请单独下载数据，并按期望的本地布局准备，然后编辑 `mini_t2i/settings.py` 指向这些数据。

**CC12M。** 预训练使用 Conceptual Captions 12M 数据。请从 [`CaptionEmporium/conceptual-captions-cc12m-llavanext`](https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext) 准备本地张量切块：

```bash
python tools/prepare_cc12m_chunks.py \
  --dataset CaptionEmporium/conceptual-captions-cc12m-llavanext \
  --out /path/to/dataset/root/cc12m_tensor_chunks
```

默认设置下，这些切块应位于 `/path/to/dataset/root/cc12m_tensor_chunks`。

对于大规模训练，公开 CC12M 图像 URL 常常失效或访问受限。一个更稳健的可选路径是使用 `img2dataset` 并行下载，然后将生成的 WebDataset shard 转换为相同的 `chunk_*.pt` 张量格式：

```bash
DATA_ROOT=/path/to/dataset/root
WORK="$DATA_ROOT/cc12m_prep"

hf download CaptionEmporium/conceptual-captions-cc12m-llavanext \
  --repo-type dataset \
  --include "*.jsonl.gz" \
  --local-dir "$WORK/meta"

python tools/cc12m_jsonl_to_parquet.py \
  --jsonl "$WORK/meta/train.jsonl.gz" \
  --out "$WORK/parquet/metadata.parquet"

img2dataset \
  --url_list "$WORK/parquet" \
  --input_format parquet \
  --url_col url \
  --caption_col caption_llava \
  --output_folder "$WORK/wds" \
  --output_format webdataset \
  --image_size 512 \
  --resize_mode center_crop \
  --encode_format jpg \
  --processes_count 16 \
  --thread_count 64 \
  --number_sample_per_shard 10000 \
  --enable_wandb False \
  --retries 2

python tools/cc12m_wds_to_chunks.py \
  --wds "$WORK/wds" \
  --out "$DATA_ROOT/cc12m_tensor_chunks"
```

**120K mix。** 微调使用来自三个公开源的 120K mix WebDataset 布局：

- [`BLIP3o/BLIP3o-60k`](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k)
- [`OpenDatasets/dalle-3-dataset`](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset)
- [`FreedomIntelligence/ShareGPT-4o-Image`](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image)

请将源数据准备为 `configs/finetune.yml` 期望的目录结构：

```bash
python tools/prepare_120k_mix_layout.py \
  --out /path/to/dataset/root/finetune_wds
```

- 默认设置下，WebDataset shards 应位于 `/path/to/dataset/root/finetune_wds`。
- 微调加载器按 `[0.06, 0.016, 0.04]` 的权重采样 BLIP3o-60K、DALL-E3 和 ShareGPT4o 数据源，近似按样本数量比例分配。
- 如果本地 shards 缺失且 `finetune_hf_streaming: true`，加载器会回退到配置在 `TrainConfig` 中的 Hugging Face 公共源。我们建议为了速度准备本地 WebDataset shards。

### CC12M 预训练

```bash
torchrun --standalone --nproc_per_node=8 \
  main.py train \
  --config-file configs/pretrain.yml \
  --output-dir "${OUT_DIR}" \
  --dataset-backend local_folder \
  --micro-batch-size 128 \
  --grad-accum-steps 1 \
  --attention-impl sdpa
```

MSCOCO-30K FID 是预训练期间的在线监控指标。请从 [`MiniT2I/evaluation-assets`](https://huggingface.co/datasets/MiniT2I/evaluation-assets/tree/main/coco) 下载 caption 文件和参考统计数据：

```bash
huggingface-cli download MiniT2I/evaluation-assets \
  --repo-type dataset \
  --include "coco/*" \
  --local-dir /path/to/evaluation/assets
```

默认设置下，该命令会创建预期的 `coco/` 资源目录。仓库中的 FID 实现使用 `pytorch-fid` 的 InceptionV3 特征提取器，并将生成图像特征与 `mscoco_fid_stats_512.npz` 进行比较。

预训练默认启用在线 FID 监控。`configs/pretrain.yml` 中设置为：

```yaml
fid_every: 20000
fid_num_samples: 30000
fid_batch_size: 64
```

训练循环会评估原始模型和 EMA 模型，并在启用 W&B 时将 FID 指标记录到 [W&B](wandb.ai)。在启动预训练前，请编辑 `mini_t2i/settings.py` 中的 `MSCOCO_CAPTION_FILE` 和 `MSCOCO_STATS_FILE`。

要对本地训练的预训练检查点手动运行相同的 FID 路径，请在 `configs/eval.yml` 中启用 `fid` 基准：

```yaml
checkpoint: /path/to/pretrained/checkpoint.pt
checkpoint_target: ema
output_dir: /path/to/eval_output

benchmarks:
  fid:
    enabled: true
    num_samples: 30000
    batch_size: 64
```

### 120K mix 微调

使用以下脚本可以复现我们的 B/32 消融级别结果。超参数可在 `configs/finetune.yml` 中调整。

```bash
torchrun --standalone --nproc_per_node=8 \
  main.py train \
  --config-file configs/finetune.yml \
  --resume-from "${PRETRAINED_CKPT}" \
  --no-auto-resume \
  --output-dir "${OUT_DIR}" \
  --micro-batch-size 64 \
  --grad-accum-steps 2 \
  --num-workers 8 \
  --attention-impl sdpa
```

## 致谢

本代码库构建于多个开源工作之上：

- [JAX MiniT2I](https://github.com/PeppaKing8/minit2i-jax) —— 本 PyTorch 迁移版本参考的原始实现。
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) 和 [Transformers](https://github.com/huggingface/transformers) —— pipeline 骨架与 FLAN-T5-Large 文本编码器。
- [PEFT](https://github.com/huggingface/peft) —— `lora/` 中使用的 LoRA 适配器。
- [`djghosh13/geneval`](https://github.com/djghosh13/geneval) —— GenEval 的提示和检测器检查点。
- [`Jialuo21/DPG-Bench`](https://huggingface.co/datasets/Jialuo21/DPG-Bench) 以及 ModelScope 的 mPLUG VQA 模型 —— DPG-Bench 评估。
- [`pytorch-fid`](https://github.com/mseitzer/pytorch-fid) —— InceptionV3 FID 特征提取。
- 来自 [`CaptionEmporium/conceptual-captions-cc12m-llavanext`](https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext)、[`BLIP3o/BLIP3o-60k`](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k)、[`OpenDatasets/dalle-3-dataset`](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset)、[`FreedomIntelligence/ShareGPT-4o-Image`](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image) 和 [`lambdalabs/naruto-blip-captions`](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) 的公开训练数据。

## 引用

```bibtex
@misc{minit2i2026,
  title  = {MiniT2I: A Minimalist Baseline for Text-to-Image Generation},
  author = {Wang, Xianbang and Zhao, Hanhong and Lu, Yiyang and Zhou, Kangyang and Ma, Linrui and He, Kaiming},
  year   = {2026},
  url    = {https://peppaking8.github.io/#/post/minit2i}
}
```
