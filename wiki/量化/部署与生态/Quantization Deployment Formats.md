---
title: Quantization Deployment Formats
aliases: [量化部署格式, 量化格式对比, Quant Formats]
tags: [model-compression, quantization, concept, deployment]
created: 2026-04-30
updated: 2026-04-30
sources: [GPTQ, AWQ, LLM.int8()]
---

## 定义

Quantization Deployment Formats 是指 LLM 量化后用于**实际推理部署**的各种格式与工具链。与理论方法（[[Post-Training Quantization]] 两步分解框架）不同，本页聚焦工程实践：各格式的兼容硬件、推理引擎、量化成本、精度表现和适用场景。

如需从学术视角了解如何根据问题选择量化方法（而非部署格式），见 [[Quantization Method Selection]]。

## 格式总览

### GPTQ（GPU Post-Training Quantization）

基于 Hessian 二阶信息的逐层权重量化（详见 [[Weight-Only Quantization]] 中的理论分析）。部署特征：

- **量化工具**：AutoGPTQ
- **典型比特宽度**：INT4（group_size=128）、INT8
- **推理引擎**：vLLM、TGI、HuggingFace Transformers、ExLlamaV2（Marlin kernel）
- **硬件**：NVIDIA GPU only（需 CUDA kernel 加速）
- **Calibration**：128 samples，耗时数小时（大模型）
- **精度**：4-bit + act-order + group_size=128 时精度优秀；低于 4-bit 退化明显

### AWQ（Activation-Aware Weight Quantization）

激活感知等价缩放保护 salient weights（详见 [[Weight-Only Quantization]] 中 AWQ 节）。部署特征：

- **量化工具**：AutoAWQ
- **典型比特宽度**：INT4（固定）
- **推理引擎**：vLLM、TGI、HuggingFace Transformers
- **硬件**：NVIDIA GPU only
- **Calibration**：16 samples，耗时数分钟
- **精度**：4-bit 下与 GPTQ 相当或略优；calibration 鲁棒性最强

### GGUF（GPT-Generated Unified Format）

llama.cpp 生态的通用量化格式。支持 CPU / GPU / 混合推理。经历三代量化算法演进（Legacy → K-Quants → I-Quants），详细技术原理见 **[[GGUF Quantization]]**。部署特征：

- **量化工具**：llama-quantize（llama.cpp 内置）
- **典型比特宽度**：1.5-bit 到 8-bit（连续覆盖）
- **推理引擎**：llama.cpp、Ollama、LM Studio、koboldcpp
- **硬件**：CPU（AVX2/AVX-512/ARM NEON）、NVIDIA GPU（CUDA）、Apple Silicon（Metal）、AMD GPU（Vulkan/ROCm）
- **Calibration**：I-quants 需要 imatrix；K-quants 无需 calibration
- **精度**：Q4_K_M 是最常用的平衡点；I-quants 在极低比特率下优于 K-quants

#### K-Quants 等级

K-quants 使用 block-wise 量化，含 super-blocks 和 sub-blocks：

| 类型 | 近似 bpw | 特点 |
|------|----------|------|
| Q2_K | ~2.63 | 极度压缩，质量损失明显 |
| Q3_K_S/M/L | ~3.07-3.50 | S/M/L 代表 sub-block 精度梯度 |
| **Q4_K_M** | **~4.58** | **最常用通用量化，质量-尺寸最佳平衡** |
| Q5_K_M | ~5.52 | 接近无损 |
| Q6_K | ~6.56 | 非常接近全精度 |
| Q8_0 | ~8.50 | 本质无损，文件大 |

#### I-Quants 等级（Importance-matrix）

使用 imatrix calibration 的非均匀比特分配：

| 类型 | 近似 bpw | 特点 |
|------|----------|------|
| IQ1_S/M | ~1.56-1.75 | 极限压缩，仅适合 70B+ 模型 |
| IQ2_XXS/XS/S/M | ~2.06-2.70 | 大模型下可用 |
| IQ3_XXS/XS/S | ~3.06-3.44 | 通常优于同 bpw 的 Q3_K |
| IQ4_XS/NL | ~4.25-4.50 | 与 Q4_K_S 竞争 |

### EXL2（ExLlamaV2 格式）

混合精度逐层量化。通过 measurement pass 分析各层敏感度，按目标 bpw 分配比特。详细技术原理见 **[[ExLlamaV2]]**。

- **量化工具**：ExLlamaV2
- **典型比特宽度**：任意连续值（如 3.0、3.5、4.0、4.65、6.5 bpw）
- **推理引擎**：ExLlamaV2、text-generation-webui（TabbyAPI）
- **硬件**：NVIDIA GPU only（CUDA）
- **Calibration**：measurement + 量化耗时数小时
- **精度**：同等平均 bpw 下通常优于 GPTQ/GGUF（得益于逐层混合精度）
- **当前状态**：项目已归档（v0.3.2），ExLlamaV3 开发中，但 EXL2 格式仍广泛使用

### BitsAndBytes（NF4 / INT8）

HuggingFace 生态的零配置量化。加载时即时量化，无需预处理。详见 [[bitsandbytes]]。

- **量化工具**：内置于 HuggingFace Transformers
- **典型比特宽度**：NF4（4-bit）、INT8
- **推理引擎**：HuggingFace Transformers（仅限）
- **硬件**：NVIDIA GPU
- **Calibration**：无需（NF4 基于正态分布假设；INT8 运行时动态分解）
- **精度**：NF4 略低于 GPTQ/AWQ（无误差补偿）；INT8 通过 outlier mixed-precision 近无损
- **核心用途**：**QLoRA 微调**——base model NF4 + LoRA adapter FP16/BF16

### FP8（8-bit Floating Point）

硬件原生浮点量化，几乎无损。

- **格式**：E4M3（推理，精度优先）、E5M2（训练，范围优先）
- **典型比特宽度**：8-bit（W8A8）
- **推理引擎**：vLLM、TensorRT-LLM
- **硬件**：NVIDIA H100/H200（Hopper）、B100/B200（Blackwell）、AMD MI300X——**无消费级 GPU 支持**
- **Calibration**：per-tensor 或 per-channel scaling，简单快速
- **精度**：几乎无损（8-bit 浮点保留远多于 INT8 的信息）
- **量化工具**：llm-compressor（vLLM 项目）、TensorRT-LLM

### AQLM（Additive Quantization for Language Models）

向量量化方法，将权重组（如 8 个）编码为多个 learned codebook entries 的加法组合。详细技术原理见 **[[AQLM]]**。

- **典型比特宽度**：2-bit（sweet spot）、3-4 bit
- **推理引擎**：HuggingFace Transformers（有限支持）
- **硬件**：GPU（需专用 kernel）
- **Calibration**：beam search + gradient fine-tuning，耗时数天
- **精度**：2-bit 下 state-of-the-art，优于 GPTQ 和 QuIP#

### QuIP#（Quantization with Incoherence Processing）

随机 Hadamard 旋转 + E₈ 格码本向量量化。详见 [[Lattice Codebooks]]、[[Rotation-based Quantization]]。

- **典型比特宽度**：2-bit、4-bit
- **精度**：首个稳定 2-bit 量化，但已被 AQLM 超越
- **生态**：主要为研究贡献，框架支持有限

## 多维对比矩阵

| 维度 | GPTQ | AWQ | GGUF | EXL2 | BitsAndBytes | FP8 | AQLM |
|------|------|-----|------|------|-------------|-----|------|
| 典型 bit | 4/8 | 4 | 1.5-8 | 任意 | 4/8 | 8 | 2-3 |
| 量化耗时 | 小时 | **分钟** | 分钟 | 小时 | **零** | 分钟 | 天 |
| CPU 推理 | 否 | 否 | **是** | 否 | 否 | 否 | 否 |
| Apple Silicon | 否 | 否 | **是(Metal)** | 否 | 否 | 否 | 否 |
| vLLM 支持 | 是 | 是 | 否 | 否 | 否 | 是 | 否 |
| 多并发服务 | **是** | **是** | 弱 | 弱 | 否 | **是** | 否 |
| 极低 bit 质量 | 弱 | 弱 | 中 | 中 | — | — | **强** |
| 4-bit 质量 | 好 | **好** | 好 | **最好** | 中 | — | 好 |
| 需 calibration | 是 | 是 | 可选 | 是 | **否** | 简单 | 是 |

## 选型决策框架

### 按部署场景

| 场景 | 推荐格式 | 理由 |
|------|----------|------|
| 数据中心 GPU 服务（H100+） | **FP8** | 几乎无损，2× 吞吐，硬件原生 |
| 数据中心 GPU 服务（A100/非 Hopper） | **AWQ INT4** 或 GPTQ INT4 via vLLM | 4× 压缩，高并发优化 |
| 单张 NVIDIA GPU 本地推理 | **GGUF**（方便）或 **EXL2**（极致质量） | GGUF 通过 Ollama 一键部署；EXL2 同 bpw 最优精度 |
| Mac / Apple Silicon | **GGUF** via Ollama/LM Studio | 唯一实际可用的选择（Metal 加速） |
| CPU-only 服务器 | **GGUF** via llama.cpp | AVX2/AVX-512 优化，无 GPU 依赖 |
| QLoRA 微调 | **BitsAndBytes NF4** | 零配置集成 HuggingFace Trainer |
| 极限压缩（2-bit） | **AQLM** | 2-bit 下质量最优；QuIP# 为备选 |
| VRAM 极度受限 + NVIDIA GPU | **EXL2** 低 bpw（如 3.0） | 混合精度最大化有限 VRAM 下的质量 |

### 按比特宽度

- **8-bit（FP8/INT8）**：几乎无损。仅 2× 压缩，适合有充足硬件但需要些许压缩的场景。FP8 优于 INT8（保留浮点语义）。
- **4-bit**：**甜蜜点**。4× 压缩，精度损失可控（尤其 QAT 或 AWQ/GPTQ + group-128）。绝大多数部署场景的最佳选择。
- **3-bit**：小模型（<14B）退化明显；大模型（70B+）仍可接受。适合 VRAM 紧张但需要较大模型的场景。
- **2-bit**：显著质量损失。仅推荐极大模型（70B+）+ AQLM/QuIP#。作为 VRAM 极度受限时的最后手段。

## GGUF 选型快速参考

对于使用 GGUF 的用户（Ollama / LM Studio），按可用内存选择：

| 可用内存 | 7B 模型推荐 | 70B 模型推荐 | 质量预期 |
|----------|------------|-------------|---------|
| 4-6 GB | Q4_K_M (~4.5 GB) | 不适合 | 好 |
| 8 GB | Q5_K_M (~5.5 GB) | IQ2_M (~20 GB) | 很好 / 有损 |
| 16 GB | Q8_0 (~7.5 GB) | Q3_K_M (~30 GB) | 近无损 / 可用 |
| 32 GB | Q8_0 | Q4_K_M (~42 GB) | 无损 / 好 |
| 64 GB+ | FP16 | Q5_K_M+ | 无损 / 很好 |

提示：imatrix GGUF（bartowski 提供）在低比特率下通常优于标准量化；Unsloth Dynamic 通过智能逐层分配在相同平均 bpw 下质量更高。

## 与理论框架的映射

| 部署格式 | 对应的 PTQ 理论方法 | 两步分解中的位置 |
|----------|-------------------|----------------|
| GPTQ | [[GPTQ]]（Hessian 自补偿） | Step 2: 量化误差补偿 |
| AWQ | [[AWQ]]（激活感知 Scaling） | Step 1: 预量化变换（Scaling） |
| EXL2 | GPTQ + Mixed Precision | Step 2 + 混合精度分配 |
| BitsAndBytes NF4 | RTN + NF4 data type | Step 2: RTN；格式：非均匀 |
| BitsAndBytes INT8 | [[LLM.int8()]]（Mixed Decomposition） | Outlier 分解 |
| FP8 | 简单缩放 + 浮点截断 | Step 1: Scaling（per-tensor/per-channel） |
| AQLM | Additive VQ + Codebook Learning | 超越两步框架（向量量化范式） |
| QuIP# | [[Rotation-based Quantization|Rotation]] + [[Lattice Codebooks|E₈ VQ]] | Step 1: Rotation + VQ 编码 |

## 趋势与展望

1. **FP4 硬件化**：NVIDIA Blackwell 原生支持 [[FP4 Quantization|MXFP4/NVFP4]]，TensorRT-LLM 和 vLLM 已在集成。4-bit 浮点可能取代 INT4 成为新标准。
2. **QAT 官方化**：Google Gemma 3 率先提供 QAT INT4 checkpoint，模型提供方直接发布"量化友好"权重可能成为趋势。
3. **动态混合精度**：Unsloth Dynamic v2.0 证明逐层智能分配比均匀量化效果更好，这一思想正在被更多工具采纳。
4. **MoE 专用量化**：DeepSeek V3/R1 的 MoE 架构催生了 expert-aware 混合精度方案（router FP32 + shared expert 高精度 + routed expert 低精度）。

详见 [[Model Quantization Ecosystem]] 了解各主流模型系列的具体量化部署现状。
