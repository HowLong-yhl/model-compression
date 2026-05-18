---
title: "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization"
aliases: [KVQuant, Hooper 2024]
source_type: paper
source_url: "https://arxiv.org/abs/2401.18079"
source_date: 2024-01-31
ingested: 2026-04-28
venue: NeurIPS 2024
authors: [Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, Amir Gholami]
affiliation: UC Berkeley
code: "https://github.com/SqueezeAILab/KVQuant"
tags: [model-compression, quantization, method, kv-cache, long-context, non-uniform, per-channel, inference]
---

## 核心要点

KVQuant 是一个面向**超长上下文**推理的 KV cache 量化框架，通过四种互补技术组合在 3-bit 下实现 <0.1 PPL 退化，使 LLaMA-7B 在单张 A100-80GB 上支持 **1M context length**，8 卡系统支持 **10M context length**。

核心洞察：Key 和 Value 的 outlier 模式根本不同——Key 有固定的 outlier channel（类似 [[Emergent Outlier Features]]），应该 per-channel 量化；而 Value 没有固定 outlier 模式。此外，RoPE 会破坏 Key 原有的 channel 稳定性，因此应在 RoPE 之前量化 Key。

## 方法详解

### 技术一：Per-Channel Key Quantization

**发现**：Key cache 的 outlier 集中在固定通道上（与 LLM 权重的 [[Emergent Outlier Features]] 类似），不同 token 的 Key 在同一通道上的分布相似。这意味着 per-channel 量化（沿 token 维度统计）比 per-token 量化更适合 Key。

**实验验证**（LLaMA-7B，WikiText-2 PPL）：

| 量化方式 | 3-bit PPL |
|---------|-----------|
| Per-token Key + Per-token Value | 5.99 |
| **Per-channel Key** + Per-token Value | **5.82** |

Per-channel 方向上 Key 的值域更紧凑，量化误差更小。

### 技术二：Pre-RoPE Key Quantization

**发现**：RoPE（Rotary Position Embedding）对 Key 的每对相邻通道施加旋转变换，旋转角度随 token 位置变化。这破坏了 Key 在通道维度上的固定 outlier 模式——RoPE 后同一通道的值域随位置变化而剧烈波动。

**方案**：在 RoPE 之前量化 Key，推理时先反量化再施加 RoPE。

**关键细节**：RoPE 仅影响 Q 和 K 的乘积（通过旋转的正交性保持内积不变），因此可以等价地将 RoPE 拆分——在 Q 端一次性应用完整 RoPE，K 端存储 pre-RoPE 值。这不改变数学结果但大幅提升了 K 的量化友好性。

### 技术三：Non-Uniform Quantization (NUQ)

传统 uniform quantization 假设数据均匀分布在 [min, max] 上，但 KV cache 的值实际呈近似正态分布，大量值集中在均值附近。NUQ 用 k-means 聚类求最优量化码本（signpost），使量化点更密地覆盖高密度区域。

**创新**：KVQuant 提出 **sensitivity-weighted k-means**——不是最小化普通 MSE，而是按每个 KV entry 对模型输出的敏感度加权。敏感度用 Fisher information 近似（对应 attention score 的大小）。

| 量化方式 | 3-bit PPL (LLaMA-7B) |
|---------|---------------------|
| Uniform | 5.82 |
| k-means (unweighted) | 5.77 |
| **Sensitivity-weighted k-means** | **5.75** |

### 技术四：Per-Vector Dense-and-Sparse Quantization

即使 NUQ 也无法完美处理 per-vector 的极端 outlier。KVQuant 对每个量化 vector 中最大的 ~1% 元素（outliers）单独以 FP16 存储，其余用低比特量化。

**存储开销**：isolating top-1% outlier 仅增加约 0.1 bit/element 的额外存储，但显著降低了剩余元素的量化范围和误差。

### 组合效果

四种技术的组合在 LLaMA-7B 和 LLaMA-2-70B 上的叠加效果：

| 方法组合 | 3-bit PPL (LLaMA-7B) | vs FP16 退化 |
|---------|---------------------|-------------|
| Baseline (per-token uniform) | 5.99 | +0.31 |
| + Per-channel Key | 5.82 | +0.14 |
| + Pre-RoPE | 5.78 | +0.10 |
| + NUQ (sensitivity-weighted) | 5.75 | +0.07 |
| + Dense-and-Sparse | **5.74** | **+0.06** |
| FP16 baseline | 5.68 | — |

**在 LLaMA-2-70B 上退化更小**——3-bit KVQuant 仅增加 ~0.03 PPL。

## 长上下文支持

KV cache 是限制长上下文推理的核心瓶颈——对于 LLaMA-7B 在 1M context length 下，KV cache 需要约 **128GB** FP16 存储（远超模型权重的 14GB）。

| 配置 | 最大 Context Length (A100-80GB) |
|------|-------------------------------|
| FP16 KV cache | ~33K |
| 4-bit KV cache | ~131K |
| 3-bit KVQuant | ~175K |
| 3-bit KVQuant + W4 weights | **~1M** |
| 3-bit KVQuant + 8× A100 | **~10M** |

## 自定义 CUDA Kernel

KVQuant 实现了自定义 CUDA kernel 以支持 NUQ 和 Dense-and-Sparse 量化的高效推理：

| 操作 | 吞吐量提升 |
|------|-----------|
| NUQ Dequantization | ~1.7× vs naive implementation |
| Prefill attention (3-bit) | 与 FlashAttention FP16 相比仅 ~15% 慢 |

反量化使用查找表（LUT）实现——3-bit 仅需 8 个 signpost，LUT 可放在 shared memory 中实现零带宽开销。

## 与其他 KV Cache 量化方法的对比

| 维度 | KVQuant | [[KIVI]] | [[TurboQuant]] |
|------|---------|-------|------------|
| 量化粒度 | Per-channel Key + Per-token Value | Per-channel Key + Per-token Value | Per-coordinate (rotation-based) |
| 量化类型 | Non-uniform (k-means) | Uniform (asymmetric) | Optimal scalar (Lloyd-Max) |
| 是否需要校准 | 需要（sensitivity weight 计算） | 不需要（tuning-free） | 不需要（data-oblivious） |
| RoPE 处理 | Pre-RoPE quantization | 未专门处理 | Random rotation 统一处理 |
| Outlier 处理 | Dense-and-Sparse isolation | Grouped + Residual | Rotation + Beta distribution |
| 目标比特 | 3-bit (<0.1 PPL 退化) | 2-bit (plug-and-play) | 2.5-3.5 bit (理论最优) |
| 主要优势 | 超长上下文支持 (10M) | 简单高效、硬件友好 | 理论保证、数据无关 |

## 局限性

- **需要 calibration 数据**：sensitivity weight 和 k-means signpost 需要离线计算，不如 KIVI 和 TurboQuant 的 tuning-free 方案灵活
- **NUQ 的 LUT 开销**：虽然 3-bit 的 LUT 很小（8 entries），但扩展到更高比特时 LUT 指数增长
- **Pre-RoPE 的 Q 端开销**：需要在 Q 端额外应用一次完整 RoPE（而非分步），增加少量计算
- **仅验证 decoder-only 模型**：未在 encoder-decoder 或 cross-attention 场景验证
- **Dense-and-Sparse 的稀疏存储**：需要额外的索引结构记录 outlier 位置

## 与已有知识的关联

### 与 LLM 量化的 Outlier 处理对比

KVQuant 的 per-channel Key quantization 发现与 [[LLM.int8()]] 首次报告的 [[Emergent Outlier Features]] 高度一致——KV cache 中的 Key 也有固定 outlier channel。区别在于处理方式：LLM.int8() 将 outlier channel 提取到 FP16 混合精度；KVQuant 用 per-channel 量化方向来适配这种模式。

### 与旋转方法的关系

[[QuaRot]] 和 [[SpinQuant]] 通过旋转消除 outlier 后量化 KV cache（W4A4KV4），是一种"预处理后 uniform 量化"路线。KVQuant 则走"不预处理但用 non-uniform 量化适配"路线。TurboQuant 将两者结合——random rotation 后用 optimal scalar quantizer。

### Pre-RoPE 发现的意义

KVQuant 的 Pre-RoPE 发现对所有 KV cache 量化方法都有参考价值——RoPE 旋转会破坏 Key 在通道维度的量化友好结构。后续 KIVI 未采用 Pre-RoPE（因为 KIVI 用 per-channel key 量化本身就部分缓解了 RoPE 影响），但 RotateKV（IJCAI 2025）等后续工作专门验证了 Pre-RoPE 的有效性。

## 引用关系

← 理论基础：[[Emergent Outlier Features]]（Key 的固定 outlier channel 模式），[[Quantization Granularity]]（per-channel vs per-token 方向的选择）
← 技术关联：Non-Uniform Quantization（k-means 码本优化），Fisher Information（sensitivity weighting）
← 相关方法：[[KIVI]]（同样使用 per-channel Key + per-token Value 方向），[[TurboQuant]]（理论 optimal 量化）
→ 贡献：首次系统研究 KV cache 量化的四个关键维度，实现 3-bit <0.1 PPL 退化
→ 应用：超长上下文推理（1M-10M tokens），单 GPU 高效 KV cache 压缩
→ 概念关联：[[KV Cache Quantization]]（本方法属于 KV cache 量化的代表作）
