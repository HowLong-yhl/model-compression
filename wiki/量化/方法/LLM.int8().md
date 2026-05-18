---
title: "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"
aliases: [LLM.int8(), LLM int8, Dettmers 2022]
source_type: paper
source_url: "https://arxiv.org/abs/2208.07339"
source_date: 2022-08-15
ingested: 2026-04-20
venue: NeurIPS 2022
authors: [Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer]
affiliation: University of Washington / Meta AI
code: "https://github.com/TimDettmers/bitsandbytes"
tags: [model-compression, quantization, method, post-training, W8A8, deployment]
---

## 核心要点

LLM.int8() 是第一个在 175B 参数规模下实现**零精度损失** INT8 推理的方法。其核心贡献是发现并刻画了大模型中的 [[Emergent Outlier Features]]，并提出了一种两阶段量化方案来应对这一挑战。

## 方法摘要

### 1. Vector-wise Quantization

传统 absmax 量化对整个 tensor 使用单一缩放因子，outlier 会"挤压"其余值的有效量化级数。LLM.int8() 改为**逐向量量化**：对隐藏状态矩阵 X 的每一行和权重矩阵 W 的每一列分别计算缩放因子，使每个内积拥有独立的归一化对，大幅降低量化误差。

反量化通过缩放因子的外积完成：`C_f16 ≈ (1 / (c_x ⊗ c_w)) · C_i32`

### 2. Mixed-Precision Decomposition

在 ~6.7B 参数以上，模型出现 [[Emergent Outlier Features]]——少数隐藏维度的激活值（幅度 > 6.0）远超正常范围（约 ±3.5）。LLM.int8() 将矩阵乘法分解为两部分：

- **Outlier 维度**（|value| ≥ 6.0）：提取出来用 **FP16** 计算
- **其余维度**（>99.9% 的值）：用 **INT8** 计算

两部分结果在 FP16 下求和得到最终输出。

### 3. 关键数值

| 指标 | 数值 |
|------|------|
| Outlier 维度数量 | ≤7（13B 模型，共数千维） |
| INT8 计算比例 | >99.9% |
| 内存节省 | 约 2x（FP16 → INT8） |
| OPT-175B 推理 | 可在 8×RTX 3090 上运行 |
| 精度损失 | 零（C4 perplexity 完全一致） |

## 关键发现：Phase Transition

在约 **6.7B 参数**处存在一个**量化退化的相变**：

- **< 6.7B**：outlier 在不同层出现在不同维度，不协调，影响约 25% 的层
- **≥ 6.7B**：outlier 突然变得高度协调——**相同维度在 100% 的层中产生 outlier**，影响约 75% 的 sequence 维度
- 将 outlier 维度置零会导致 **600-1000% 的 perplexity 退化**，并使 top-1 attention 概率质量下降超过 20%

这一相变是传统 INT8 量化在大模型上失败的根本原因，也是 mixed-precision decomposition 的动机。

## 实验结果

### Perplexity (C4 validation)

| 模型 | FP32 | Absmax INT8 | Vector-wise INT8 | **LLM.int8()** |
|------|------|-------------|-------------------|-----------------|
| 2.7B | 14.43 | 15.11 | 14.98 | **14.44** |
| 6.7B | 13.30 | 14.59 | 14.13 | **13.24** |
| 13B | 12.45 | 19.08 | 16.48 | **12.45** |

13B 时 naive absmax 退化 53%，而 LLM.int8() 完全无损。

### 推理延迟 (BLOOM-176B, ms/token)

| 配置 | Batch=1 | Batch=8 |
|------|---------|---------|
| bf16, 8×A100 | 239 ms | 32 ms |
| LLM.int8(), 8×A100 | 253 ms | 34 ms |
| LLM.int8(), 4×A100 | 246 ms | 33 ms |

端到端延迟仅增加约 5-6%，且使用更少 GPU 时由于减少了通信开销，延迟反而可能更低。

## 实现与集成

通过 [[bitsandbytes]] 库开源实现，已深度集成到 HuggingFace Transformers 中，只需 `load_in_8bit=True` 即可使用。应用于 feed-forward 层和 attention projection 层，不涉及 attention 函数本身。

## 局限性

- 仅针对 INT8，未探索更低精度（INT4 等）
- Mixed-precision decomposition 在小模型上有显著速度开销（768 维时仅 0.14× FP16 速度）
- 不涉及训练阶段的量化
- INT8 本身不是深度学习的理想数据类型，方法是在硬件限制下的工程折衷

## 与已有知识的关联

- 本文发现的 [[Emergent Outlier Features]] 是后续 [[SmoothQuant]] 和 [[AWQ]] 的核心动机
- 与 [[SmoothQuant]] 的区别：LLM.int8() 通过运行时混合精度分解处理 outlier，SmoothQuant 通过离线等价变换"平滑"掉 outlier
- 与 [[GPTQ]] 的区别：LLM.int8() 关注 W8A8（权重+激活都量化），GPTQ 关注 weight-only 量化到更低精度（3-4 bit）

## 引用信息

原始来源：`raw/LLM.int8().md`（待录入）
