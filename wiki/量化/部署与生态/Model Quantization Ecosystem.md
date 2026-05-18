---
title: Model Quantization Ecosystem
aliases: [模型量化生态, LLM 量化生态, 量化版本汇总]
tags: [model-compression, quantization, concept, deployment]
created: 2026-04-30
updated: 2026-04-30
sources: [GPTQ, AWQ, LLM.int8()]
---

## 定义

Model Quantization Ecosystem 记录主流开源 LLM 系列的量化部署现状——官方发布了哪些量化版本、社区提供了哪些补充、以及各模型架构对量化的特殊影响。本页是 [[Quantization Deployment Formats]] 的实践伴侣。

---

## Qwen 系列

### 官方量化（Alibaba / Qwen 团队）

Qwen 团队在 HuggingFace `Qwen/` namespace 下发布全系列官方量化模型，命名规范一致：

```
Qwen/{ModelFamily}-{Size}-{Variant}-{QuantFormat}
```

示例：`Qwen/Qwen3-32B-AWQ`、`Qwen/Qwen3-235B-A22B-GPTQ-Int4`、`Qwen/Qwen3-8B-GGUF`、`Qwen/Qwen3-14B-FP8`

MoE 模型含激活参数标注：`-A{ActiveParams}B`（如 `235B-A22B`、`30B-A3B`）

#### 格式覆盖

| 格式 | 比特宽度 | 工具链 | 说明 |
|------|----------|--------|------|
| GPTQ-Int4 | INT4 | AutoGPTQ | 所有 Instruct 模型均提供 |
| GPTQ-Int8 | INT8 | AutoGPTQ | 较大模型提供 |
| AWQ | INT4 | AutoAWQ | 固定 4-bit，命名无后缀 |
| GGUF | Q2_K–Q8_0, F16 | llama.cpp | 单 repo 含多量化等级 |
| FP8 | W8A8 | llm-compressor | block size 128 细粒度量化 |

#### 演进趋势

- **Qwen2.5**（2024.09）：以 GPTQ + AWQ 为主，FP8 部分覆盖
- **Qwen3**（2025.04）：全面扩展，GPTQ + AWQ + GGUF + FP8 四格式齐发
- **Qwen3.5**（2025 年底）：确认 MXFP4 支持（TensorRT-LLM + vLLM 集成）
- **趋势**：官方覆盖日益全面，社区量化的必要性降低

### 社区量化

| 量化者 | 主要格式 | 特点 |
|--------|----------|------|
| **bartowski** | GGUF, EXL2 | imatrix calibration，覆盖完整 quant ladder |
| **unsloth** | GGUF | Dynamic v2.0 逐层智能分配，同 bpw 下质量更高 |
| **QuantTrio** | AWQ, FP8, GPTQ-Int8 | 填补官方空缺（如 VL 模型 AWQ） |
| **Intel** | INT4 (AutoRound) | Intel 硬件优化方案 |
| **MLX community** | MLX 4-bit | Apple Silicon 原生 |

---

## DeepSeek 系列

### 官方发布

DeepSeek 的量化策略与其他模型迥异——**原生 FP8 训练**是核心特色。

#### 原生 FP8 训练（V3/R1）

DeepSeek-V3（671B 总参数，37B 激活）是首个大规模开源 FP8 原生训练模型：

- **前向传播**：权重和激活使用 **E4M3** FP8 进行 GEMM 计算
- **量化粒度**：per-tile（128×128）online scaling，非 per-tensor
- **累积精度修复**：NVIDIA Tensor Core FP8 累积仅 14-bit 精度，DeepSeek 通过选择性 CUDA core 高精度累积修复 ~2% 误差
- **发布格式**：直接发布 FP8 权重（业界首例）；社区后转为 BF16 以兼容更多引擎

#### DeepSeek-V3.1 与 UE8M0

V3.1 引入 **UE8M0（Unsigned 8-bit Exponent, 0-bit Mantissa）** 作为 scaling factor 格式——scale 只能是 2 的幂，对齐 OCP MX 标准和 Blackwell 硬件。去量化简化为 bit shift 而非乘法。

#### R1 Distilled 模型

R1 蒸馏模型（Qwen-1.5B/7B/14B/32B、Llama-8B/70B）为 dense 架构，标准量化工具直接适用。

### 社区量化

| 方案 | 作者/团队 | 核心思想 | 目标硬件 |
|------|----------|----------|----------|
| **1.58-bit Dynamic** | Unsloth | 关键层 4-6 bit + MoE 专家 1.58-bit + router FP32 | 多卡/大内存 |
| **Block-INT8 / Channel-INT8** | 美团 | FP8→BF16→INT8，针对 A100 | NVIDIA A100（无 FP8） |
| **W4AFP8** | novita | Dense 层 FP8 + MoE 专家 INT4 | GPU (SGLang) |
| **KTransformers** | kvcache-ai | MLA+shared expert 在 GPU (FP8) + MoE routed expert 在 CPU (GGUF) | 消费级 GPU + CPU |
| **标准 GGUF** | bartowski | imatrix 全 quant ladder | CPU / hybrid |

#### 美团 INT8 部署方案

解决 A100 无 FP8 Tensor Core 的问题：

- Block-wise INT8（128×128 tile）：32 张 A100 上吞吐提升 33%（vs BF16）
- Channel-wise INT8：吞吐提升 50%
- 精度：GSM8K / MMLU 上本质无损
- 已合入 SGLang 推理框架

### MoE + MLA 架构的特殊挑战

DeepSeek 的 MoE + [[DeepSeek-V2-MLA|MLA]] 架构给量化带来独特问题：

1. **Router 敏感性**：路由权重极度敏感，微小扰动可导致 routing collapse（所有 token 路由到同一 expert）。因此 Unsloth 保持 router 为 FP32。
2. **Expert 异质性**：不同 expert 权重分布差异大，均匀量化 suboptimal。EAQuant、MoPEQ 等研究发现 expert 间量化敏感度差异显著。
3. **Shared vs Routed Expert**：Shared expert（始终激活）承载更关键信息，需更高精度；Routed expert（条件激活）可容忍更激进量化。
4. **MLA 误差放大**：MLA 将 KV 压缩到低秩 latent（dim 512），本身是有损压缩；在其上再量化会导致误差叠加放大，尤其 up-projection 矩阵敏感。
5. **稀疏激活模式**：每次仅激活 37B/671B 参数，calibration 数据难以覆盖所有 expert，低频 expert 可能校准不足。

---

## Gemma 系列

### 官方量化（Google）

#### Gemma 3 QAT INT4（2025.03）

Google 为 Gemma 3 全系列（1B/4B/12B/27B）发布官方 **QAT（Quantization-Aware Training）** INT4 checkpoint——这是主流模型厂商中首个大规模 QAT 发布。

**QAT 方法详情**：
- 训练过程中模拟 INT4 weight per-channel 量化噪声，使模型学会适应 4-bit 精度损失
- 发布的是 `qat-int4-unquantized` checkpoint（权重仍为 BF16 存储，但已具备 INT4 鲁棒性）
- 社区可在此基础上转换为 GGUF / GPTQ / AWQ，质量优于直接量化原始 BF16 模型

**VRAM 节省**：
- 27B：BF16 ~54 GB → QAT INT4 ~14.1 GB（单张 RTX 4090 可运行）
- 12B：QAT INT4 ~8 GB
- 4B/1B：极轻量

**发布位置**：HuggingFace `google/gemma-3-{1b,4b,12b,27b}-it-qat-int4-unquantized`

#### Gemma 2

未发布官方量化版本。所有量化由社区提供。

### 社区量化

| 量化者 | 格式 | 说明 |
|--------|------|------|
| **unsloth** | GGUF | Dynamic v2.0，含标准 Gemma 3 和 QAT 版本 |
| **lmstudio-community** / bartowski | GGUF | imatrix 量化，多种 quant level |
| **hugging-quants** | AWQ | HuggingFace 官方量化团队提供 |
| **shuyuej** | GPTQ | Gemma 2 9B/27B GPTQ 版本 |

**最佳实践**：对 Gemma 3，始终优先使用 Google 的 QAT checkpoint 作为量化源（而非原始 BF16），因为 QAT 权重在任何下游格式（GGUF/AWQ/GPTQ）中量化后质量都更优。

### 架构特性对量化的影响

**Gemma 2 的问题**：
- **Logit Soft-Capping**（`tanh(logits/cap) × cap`）：早期 llama.cpp 不支持此操作，GGUF 产出乱码。需推理引擎显式实现。
- 交错 Local/Global Attention（4096 滑动窗口 + 全局注意力）

**Gemma 3 的改进**：
- 移除 Soft-Capping，改用 QK-Norm 稳定注意力——量化兼容性大幅提升
- 多模态（SigLIP 视觉编码器）：通常仅量化语言模型部分
- 128K 上下文：长上下文推理时 KV cache 量化变得重要

---

## 社区量化者生态

| 量化者                    | 活跃度       | 主要格式                | 特色                                        |
| ---------------------- | --------- | ------------------- | ----------------------------------------- |
| **bartowski**          | 极高        | GGUF, EXL2          | imatrix calibration，完整 quant ladder，覆盖率最广 |
| **unsloth**            | 高         | GGUF                | Dynamic v2.0 逐层智能精度分配，低 bit 下质量领先         |
| **TheBloke**           | 降低（新模型减少） | GGUF, GPTQ, AWQ     | 历史上最多产，但新一代模型官方已自行提供                      |
| **QuantTrio**          | 中         | AWQ, FP8, GPTQ-Int8 | 填补官方空白                                    |
| **lmstudio-community** | 中         | GGUF                | 为 LM Studio 生态优化                          |
| **hugging-quants**     | 中         | AWQ                 | HuggingFace 官方量化项目                        |

### Unsloth Dynamic Quantization

值得特别关注的社区创新：

- **核心思想**：分析每层对量化的敏感度，为关键层分配更高精度，非关键层更低精度
- **效果**：同等平均 bpw 下，Dynamic v2.0 通常优于标准 GGUF 量化（如 UD-Q4_K_XL 优于标准 Q5_K_M）
- **原理与 EXL2 类似**，但面向 GGUF 生态（CPU/混合推理）而非 GPU-only
- DeepSeek-R1 的 1.58-bit Dynamic 是最极端的案例：88% 参数 1.58-bit + 关键层 4-6 bit + router FP32，720 GB → 131 GB

---

## 三大系列量化策略对比

| 维度 | Qwen | DeepSeek | Gemma |
|------|------|----------|-------|
| 官方量化投入 | **全面**（4 格式全覆盖） | 中（原生 FP8 为主） | 中（QAT INT4 为亮点） |
| 量化理念 | PTQ 全格式覆盖 | 训练即量化（FP8 native） | QAT（训练时量化感知） |
| 最低比特官方支持 | INT4 (GPTQ/AWQ) | FP8 (~8-bit) | INT4 (QAT) |
| 社区依赖度 | 低（官方覆盖广） | 高（低 bit 全靠社区） | 中（GGUF 靠社区） |
| 架构挑战 | 低（标准 dense/MoE） | **高**（MoE+MLA） | 中（soft-capping 历史问题） |
| 独特贡献 | 命名规范示范 | FP8 训练范式 | QAT 公开 checkpoint 范式 |

---

## 选型实践建议

### 我有 H100/H200 集群

→ 直接使用 **FP8** 版本。Qwen 和 DeepSeek 官方均提供 FP8，几乎无损，吞吐最高。

### 我有 A100 集群（无 FP8 Tensor Core）

→ Qwen：官方 **GPTQ-Int4** 或 **AWQ** via vLLM
→ DeepSeek：美团 **Channel-INT8**（SGLang 集成，50% 吞吐提升 vs BF16）

### 我有单张 RTX 4090 / 5090（24 GB）

→ Qwen-32B / Gemma-27B：**AWQ INT4** (~16-18 GB) via vLLM 或 GGUF Q4_K_M via Ollama
→ Gemma-27B：优先选 **QAT 版本的 GGUF**（如 `gemma-3-27b-it-qat-GGUF` Q4_K_M）

### 我想在 Mac 上运行

→ **GGUF** via Ollama / LM Studio（Metal 加速）
→ Qwen 系列另有 **MLX** 4-bit 选项

### 我想运行 DeepSeek-R1（671B）

→ 多卡：官方 **FP8**（需 8×H100 80GB 或 16×A100+美团 INT8）
→ 极限压缩：Unsloth **1.58-bit Dynamic**（~131 GB，多卡或大内存 CPU）
→ 消费级：**KTransformers** hybrid（14-24 GB GPU + 大内存 CPU，单用户速度可用）

### 我要微调

→ **BitsAndBytes NF4** + QLoRA（HuggingFace 生态无缝集成）

详细格式对比和通用决策框架见 [[Quantization Deployment Formats]]。
