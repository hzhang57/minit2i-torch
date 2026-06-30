# MiniT2I 代码结构分析

以下是 `minit2i-torch` 仓库的主要代码组成与模块职责说明，帮助快速理解项目架构和功能分区。

## 1. 顶层结构

```text
.
├── LICENSE
├── README.md
├── README_CN.md
├── code_tree.md
├── main.py
├── requirements.txt
├── assets/
├── configs/
├── diffusers/
├── lora/
├── mini_t2i/
└── tools/
```

### 顶层文件说明
- `LICENSE`：MIT 许可证文件。
- `README.md`：英文项目介绍与使用说明。
- `README_CN.md`：中文说明文档。
- `code_tree.md`：本文件，项目结构与模块分析。
- `main.py`：统一入口脚本，支持 `train` 和 `eval` 两个子命令。
- `requirements.txt`：Python 依赖列表。
- `assets/`：项目展示图片等静态资源。

## 2. 关键目录说明

### `configs/`
存放训练、微调和评估的 YAML 配置文件。
- `pretrain.yml`：预训练配置。
- `finetune.yml`：微调配置。
- `eval.yml`, `eval_hf_b16.yml`, `eval_hf_l16.yml`：评估配置。

这些配置文件定义了模型参数、数据路径、训练超参、评估基准等。

### `diffusers/`
包含自定义的 Hugging Face Diffusers pipeline 实现，用于推理与 Hub 加载。
- `mmdit.py`：定义 Diffusers 风格的模型 config 和 diffusion model。
- `pipeline.py`：自定义 `MiniT2ITextToImagePipeline`，实现文本到图像的推理接口。

这个目录的代码主要作用是让 MiniT2I 模型可以以 Diffusers 格式加载、保存和使用。

### `lora/`
包含 LoRA 适配训练与采样脚本。
- `train_lora.py`：LoRA 适配器训练入口。
- `sample_lora.py`：加载 LoRA 适配器并进行采样。

该目录专注于将 MiniT2I 与 PEFT LoRA 结合，用于轻量域适应和下游调优。

### `mini_t2i/`
仓库核心代码目录，包含训练、评估、模型、数据、工具函数等。

```text
mini_t2i/
├── __init__.py
├── config.py
├── data.py
├── diffusion.py
├── dpg_data.py
├── eval_pipeline.py
├── evaluation/
├── fid.py
├── model.py
├── settings.py
├── train.py
├── utils.py
├── visualize.py
├── datasets/
└── scripts/
```

#### `mini_t2i/__init__.py`
空文件或包初始化文件，保证 `mini_t2i` 可作为 Python 包导入。

#### `mini_t2i/settings.py`
全局路径与资源配置。
- 定义数据根目录、输出根目录、检查点根目录、评估资源根目录。
- 定义 Hugging Face token、W&B token 等路径。
- 计算派生路径，例如 `CC12M_TENSOR_ROOT`、`GENEVAL_METADATA` 等。
- 提供 `expand_setting` 函数，用于读取配置文件中的路径别名。

#### `mini_t2i/config.py`
训练配置类 `TrainConfig`。
- 使用 dataclass 定义默认训练参数、模型结构参数、数据参数、优化参数。
- 支持从 YAML 文件加载并扩展路径。
- 提供 `resolve` 方法创建输出目录。

#### `mini_t2i/data.py`
训练数据加载和校验逻辑。
- `check_training_data`：检查预训练 tensor chunks 或微调 WebDataset 的存在性。
- `make_loader`：构建 `DataLoader`。
- `collate`：批处理拼接函数。

#### `mini_t2i/datasets/`
数据集后台子模块。
- `interface.py`：根据 `TrainConfig` 选择数据提供者。
- `finetune.py`：微调 WebDataset 混合流实现。

`datasets` 目录负责不同数据源的统一访问：
- `local_folder`：基于本地 `chunk_*.pt` 张量块的数据流。
- `finetune_wds`：基于本地 WebDataset `.tar` 的混合流式读取和文本编码。

#### `mini_t2i/model.py`
核心模型实现。
- 定义 `RMSNorm`、`SwiGLU`、`TimestepEmbedder`、各类旋转位置编码函数等基础组件。
- 实现 `PlainTextBlock`、`DoubleStreamBlock` 等 Transformer 风格模块。
- 实现 `MMJiTB32Text2` 主模型结构。

这是 MiniT2I 在 PyTorch 侧的最关键代码，负责图像块嵌入、文本与图像双流交互、去噪器前向计算。

#### `mini_t2i/diffusion.py`
训练损失与采样逻辑。
- `training_loss`：基于流匹配的 velocity 损失计算。
- `euler_sample`：基于 Euler ODE 采样推理。

该模块封装了 MiniT2I 的训练目标和采样流程，是训练与生成的核心算法部分。

#### `mini_t2i/fid.py`
FID 评估工具。
- 使用 `pytorch_fid` 的 InceptionV3 提取特征。
- 支持从图像或特征计算 FID。
- 支持加载 MSCOCO caption 文件。

#### `mini_t2i/train.py`
训练入口与训练循环。
- 初始化分布式训练环境。
- 构建模型、文本编码器、优化器、EMA 模型。
- 加载数据、计算 loss、更新权重、保存 checkpoint。
- 支持在线可视化、FID 评估和 W&B logging。

该文件连接配置、模型、数据和训练逻辑，是最重要的训练脚本。

#### `mini_t2i/eval_pipeline.py`
评估入口与 benchmark 调度。
- 读取评估配置文件。
- 支持 FID、GenEval、DPG-Bench 等评估流程。
- 通过子进程调用训练或脚本生成基准图像。
- 支持本地 checkpoint 和 Hugging Face Hub 模型。

它是仓库中评估实验编排的核心模块。

#### `mini_t2i/utils.py`
通用工具函数。
- 分布式辅助：`rank()`、`world_size()`、`is_main()`。
- 随机种子、类型转换、EMA 更新、原子保存等实用函数。

#### `mini_t2i/visualize.py`
图像保存工具。
- `save_grid`：将张量图像拼接并保存为网格图。

#### `mini_t2i/dpg_data.py`
DPG-Bench 评估相关数据处理（高级评估数据管理，非核心训练）。

### `mini_t2i/evaluation/`
外部 benchmark 集成。
- `geneval/`：GenEval 评估脚本和结果摘要代码。

该目录用于将 MiniT2I 生成结果与第三方评估工具连接。

### `mini_t2i/scripts/`
评估辅助脚本。
- `evaluate_dpg_bench.py`：DPG-Bench 评估脚本。
- `generate_benchmark_images.py`：生成 benchmark 图片。

## 3. `tools/` 目录说明

```text
tools/
├── cc12m_jsonl_to_parquet.py
├── cc12m_wds_to_chunks.py
├── prepare_120k_mix_layout.py
└── prepare_cc12m_chunks.py
```

这些脚本用于训练数据准备：
- `prepare_cc12m_chunks.py`：将 CC12M 数据集处理成本地 `chunk_*.pt` 张量块。
- `cc12m_jsonl_to_parquet.py`：从 JSONL 格式转换成 Parquet。
- `cc12m_wds_to_chunks.py`：从 WebDataset shards 生成预训练 tensor chunks。
- `prepare_120k_mix_layout.py`：准备 120K mix WebDataset 的目录布局。

## 4. `lora/` 目录分析

```text
lora/
├── sample_lora.py
└── train_lora.py
```

该目录实现 MiniT2I 的 LoRA 适配方案：
- `train_lora.py`：使用 Hugging Face Accelerate 和 PEFT 训练 LoRA adapter。
- `sample_lora.py`：加载适配器并对输入 prompt 生成图像。

作用是支持轻量化领域适应，不修改原始 MiniT2I 基模型的参数。

## 5. `diffusers/` 目录分析

```text
diffusers/
├── mmdit.py
└── pipeline.py
```

功能集中在将 MiniT2I 封装为 Diffusers pipeline：
- `mmdit.py`：定义 MiniT2I 在 Diffusers 下的模型配置与调度器。
- `pipeline.py`：实现从 Hugging Face Hub 加载、保存与推理的接口。

如果你希望直接使用 `DiffusionPipeline.from_pretrained` 加载 MiniT2I，这里是关键代码。

## 6. 代码组成总结

- 训练主线：`main.py -> mini_t2i/train.py -> mini_t2i/model.py / mini_t2i/diffusion.py / mini_t2i/data.py`
- 评估主线：`main.py eval -> mini_t2i/eval_pipeline.py -> mini_t2i/evaluation/ / mini_t2i/fid.py`
- 数据处理：`tools/` + `mini_t2i/datasets/`
- Diffusers 推理：`diffusers/` + `lora/`（适配器）
- 配置与路径管理：`mini_t2i/config.py` + `mini_t2i/settings.py`
- 通用工具：`mini_t2i/utils.py` + `mini_t2i/visualize.py`

## 7. 主要逻辑流

1. `python main.py train --config-file configs/pretrain.yml`:
   - 通过 `mini_t2i.train` 开始。
   - 读取 `TrainConfig`，初始化模型与数据。
   - 迭代训练、保存 checkpoint、可视化样本、计算 FID。

2. `python main.py eval --config-file configs/eval.yml`:
   - 通过 `mini_t2i.eval_pipeline` 进入。
   - 加载评估配置，调度 FID、GenEval、DPG-Bench。
   - 生成图像、调用第三方评估脚本、写入结果。

3. `python lora/train_lora.py ...`:
   - 使用 Diffusers/PEFT 训练 LoRA adapter。

4. `python lora/sample_lora.py ...`:
   - 加载适配器和 MiniT2I 基模型，执行文本到图像采样。

## 8. 适合关注的重点文件

- `mini_t2i/train.py`：训练流程核心。
- `mini_t2i/model.py`：模型结构核心。
- `mini_t2i/diffusion.py`：损失与采样算法。
- `mini_t2i/eval_pipeline.py`：评估调度逻辑。
- `diffusers/pipeline.py`：Diffusers 推理入口。
- `lora/train_lora.py`：LoRA 模型适配训练。

---

该文件旨在提供一个快速、可读的中文代码结构指南，方便你理解仓库的功能分区与关键实现路径。