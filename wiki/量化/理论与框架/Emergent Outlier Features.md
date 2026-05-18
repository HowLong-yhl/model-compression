---
title: Emergent Outlier Features
aliases: [Outlier Features, 涌现异常特征, Activation Outliers]
tags: [model-compression, quantization, concept, theory]
created: 2026-04-20
updated: 2026-04-20
sources: [LLM.int8(), SmoothQuant, AWQ]
---

## 定义

Emergent Outlier Features 是指在大规模 Transformer 语言模型中出现的一种**高度系统性的极端幅度激活值**。它们出现在特定的隐藏维度中，幅度可达正常值的 **20-100×**（LLM.int8() 报告 ~60 vs. ±3.5 即 ~20×；SmoothQuant 报告可达 ~100×，因模型和层而异），由 [[LLM.int8()]]（Dettmers et al., 2022）首次发现和刻画。

## 核心特征

Outlier features 具有三个定义性特征（需同时满足）：

1. **幅度 ≥ 6.0**（正常范围约 ±3.5）
2. 影响 **≥ 25% 的 Transformer 层**
3. 影响 **≥ 6% 的 sequence 维度**

关键性质：它们是 **dimension-specific** 的——相同的少数隐藏维度在所有层、所有 token、所有序列中持续产生 outlier。这种"通道固定"特性使得 per-channel 方法特别有效。

## Phase Transition（相变）

在约 **6.7B 参数**处存在一个尖锐的相变：

| 指标 | < 6.7B | ≥ 6.7B |
|------|--------|--------|
| 受影响的层 | ~25% | **100%** |
| 受影响的 sequence 维度 | ~35% | **~75%** |
| 维度间协调性 | 不同层不同维度 | **所有层相同维度** |

相变前 outlier 是零散的；相变后变得高度协调，构成一种"涌现"现象。

## 术语辨析

不同论文对同一现象使用了不同术语：

- **Outlier features / Activation outliers**（LLM.int8(), SmoothQuant）：从激活分布角度命名
- **Salient weight channels**（AWQ）：从权重重要性角度命名——大激活值的通道对应的权重量化误差会被放大，故这些通道的权重是"salient"的
- **Outlier channels**（通用叙述）：强调这是 channel-wise 现象

这三个术语描述的是**高度重叠甚至相同的通道集合**：LLM.int8() 发现某些通道持续产生大幅激活 → SmoothQuant 对这些通道做 smoothing → AWQ 将其识别为 salient weights 需要保护。

## 对量化的影响

Outlier 是 LLM 量化的**核心挑战**：

- **Per-tensor 量化**：outlier 支配缩放因子，将所有正常值压缩到仅 2-3 个有效量化级，导致灾难性精度损失
- 13B 模型上 naive INT8 absmax 量化：perplexity 从 12.45 退化到 **19.08**（+53%）
- 将 outlier 维度置零会导致 **600-1000% perplexity 退化** 和 top-1 attention 概率质量下降 >20%

Outlier 维度数量随模型规模增长，但**具体数值因阈值定义而异**：LLM.int8() 使用严格三重标准（幅度≥6.0 + 影响≥25%层 + ≥6%序列）时，13B 模型仅 ≤7 维；若放宽为仅幅度超出阈值的维度计数，则 6B 约 15 维，13B 约 60 维，66B 约 95 维。无论哪种定义，占总隐藏维度的比例都很小（<2%）。

## 各方法的应对策略

| 方法 | 策略 | 原理 |
|------|------|------|
| [[LLM.int8()]] | Mixed-precision decomposition | Outlier 维度用 FP16 计算，其余 INT8，运行时分解 |
| [[SmoothQuant]] | 等价变换平滑 | 离线将 outlier 的"难度"迁移到权重，使激活变得均匀 |
| [[AWQ]] | 激活感知缩放 | 利用 outlier 通道信息识别 salient weights 并保护 |
| [[GPTQ]] | Hessian 误差补偿 | 不直接处理 outlier，而是通过二阶信息优化量化后的全局误差 |
| [[Rotation-based Quantization]] | 正交旋转分散 | 将 outlier 均匀分散到所有维度，使分布变均匀（最彻底的消除方式） |

## 对 Attention 的影响

Outlier features 对 attention 机制至关重要：

- 出现后 attention 变得**极度稀疏**（99% sparsity）——这正是 [[Attention Sparsity]] 的物理根源之一
- FFN 层的可剪枝性从 30% 下降到 **<5%**
- 它们编码了 attention 机制用于实现选择性、尖锐注意力模式的关键信息

Outlier 与 attention sparsity 的关系是双向的：outlier features 驱动了注意力的稀疏化，而这种稀疏性反过来支撑了 [[Token Eviction]]（移除低权重 token 的 KV）和 [[KV Cache Quantization|Value 量化]]（低权重 token 的量化误差对输出无影响）的可行性。

## 成因（推测）

论文尚未完全解释 outlier 的成因。推测与 Transformer 训练动态有关——在某个临界规模后，模型"学会"利用少数固定维度来编码全局信息（类似 positional 或 structural encoding）。这仍是一个开放研究问题。

## KV Cache 中的 Outlier

Outlier features 不仅存在于前向传播的激活中，也直接体现在 KV cache 里——Key cache 继承了同样的固定 outlier channel 模式。[[KVQuant]] 和 [[KIVI]] 独立发现 Key 的 outlier 集中在固定通道上且跨 token 稳定，因此 Key 应 per-channel 量化。Value cache 则没有这种固定模式。此外 [[KVQuant]] 发现 RoPE（旋转位置编码）会破坏 Key 的通道稳定性，提出 Pre-RoPE Key Quantization 来回避此问题。详见 [[KV Cache Quantization]]。

## 扩散模型中的 Outlier

扩散模型（Diffusion Models）中也存在类似的 outlier 现象，但具有**时间步依赖性**——不同去噪时间步 t 下，outlier channel 的位置和幅度会发生变化，这比 LLM 中固定的 outlier channel 更难处理。PTQ4DiT（NeurIPS 2024）首次在 DiT 架构中识别了这一特性，提出了时间步感知的 channel-wise 重参数化来应对。MixDQ（ECCV 2024）则发现序列首个 token（BOS）的激活幅度远大于其余 token，是另一种形式的"位置依赖 outlier"。详见 [[Diffusion Model Quantization]]。
