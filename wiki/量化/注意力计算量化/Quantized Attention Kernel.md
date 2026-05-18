---
title: Quantized Attention Kernel
aliases: [量化注意力计算, Quantized Attention, Attention Quantization, 注意力量化加速]
tags: [model-compression, quantization, concept, attention-kernel, inference]
created: 2026-04-29
updated: 2026-04-29
sources: [SageAttention, SageAttention2, SageAttention3]
---

## 定义

Quantized Attention Kernel 是指将 attention 计算中的矩阵乘法（QK^T 和 PV）从 FP16/BF16 量化到低比特（INT8/INT4/FP8/FP4）来加速推理的技术。与 [[KV Cache Quantization]]（压缩**存储**在 cache 中的 KV 向量以降低内存）不同，quantized attention kernel 压缩的是**计算过程本身**以提升吞吐量。两者完全正交，可以同时使用。

## 为什么要量化 Attention 计算

在长序列场景下，attention 的 O(N²) 复杂度使其成为推理的主要瓶颈——线性层是 O(N)。现代 GPU 的低比特 Tensor Core 吞吐远高于 FP16：

| GPU | FP16 | INT8 | FP8 | FP4 (microscaling) |
|-----|------|------|-----|-----|
| RTX4090 | ~165 TOPS | ~660 TOPS | ~330 TOPS | — |
| RTX5090 | ~200 TOPS | — | ~419 TOPS | **~1600 TOPS** |
| H100 | ~990 TOPS | — | ~1980 TOPS | — |

量化 attention 的矩阵乘法可以直接利用这些低比特计算单元，理论加速 2-8×。

## 核心挑战

### 挑战一：K/Q 的 Outlier

K 矩阵继承了 [[Emergent Outlier Features]]——特定通道持续产生极端值。直接量化导致灾难性精度损失（CosSim 从 100% 降到 <30%）。Q 矩阵也存在 block-wise outlier（尽管不如 K 明显）。

**解决方案演进**：

| 方法 | 技术 | 效果 |
|------|------|------|
| [[SageAttention]] | Smooth K（减去 token-wise 均值） | CosSim 30.60% → 99.31% |
| [[SageAttention2]] | Smooth Q+K（双向减均值） | CosSim 80.04% → 99.46% |
| FlashAttention3 | 无 smoothing | CosSim 26.76%（视频模型崩溃） |

Smooth K 的数学基础：$\text{softmax}(QK^T) = \text{softmax}(Q(K-\bar{K})^T)$，softmax 对 K 的常数偏移不变。Smooth Q 则需要一个低成本 GEMV 来计算修正项 $\Delta S$。

### 挑战二：P 矩阵的特殊分布

P = softmax(S) 的值域在 [0, 1]，且高度稀疏（>95% 接近零，见 [[Attention Sparsity]]）。这种分布对量化不友好：

- INT8 对 P 效果极差（P 本质非均匀，CosSim 77.05%）
- FP8 E4M3 效果好（适配 P 的对数分布，CosSim 99.44%）
- FP4 需要 two-level scaling：先 per-token 归一化到 [0, 448×6]，再 microscaling 量化

### 挑战三：累加器精度

GPU Tensor Core 的 FP8 累加器实际只有 **FP22 精度**（1+8+13 bits），而非声称的 FP32。[[SageAttention2]] 首次发现这一问题并提出 two-level accumulation（用真 FP32 buffer 定期收集，0% overhead）。

## 方法演进

```
2024.10  SageAttention (ICLR 2025)     ← INT8 QK^T + FP16 PV, smooth K
         峰值 341 TOPS, ~2.1× FlashAttn2

2024.11  SageAttention2 (ICML 2025)    ← INT4 QK^T + FP8 PV, smooth Q+K, per-thread 量化
         峰值 481 TOPS, ~3× FlashAttn2, ~4.5× xformers
         匹配 FlashAttn3(FP8) 速度但精度远优

2025.05  SageAttention3 (NeurIPS 2025) ← FP4 microscaling QK^T + FP4 PV, two-level scaling
         峰值 1038 TOPS, **5× FlashAttn2**
         首个可训练 8-bit attention (SageBwd)
```

## Speed-Accuracy 全景

| 方法 | TOPS (RTX5090) | CosSim |
|------|---------------|--------|
| FlashAttention2 (FP16) | 214 | 100.000% |
| [[SageAttention]] (INT8) | 479 | 99.996% |
| [[SageAttention2]] (INT4+FP8) | 643 | 99.995% |
| FlashAttention3 (FP8) | — | 98.570% |
| [[SageAttention3]] (FP4) | **1038** | **99.551%** |

SageAttention 系列在每一代都实现了"更快且更准"——这得益于 smoothing 技术持续消除 outlier，使低比特量化的精度损失极小。

## INT vs FP 的选择

一个贯穿 SageAttention 系列的发现：**smooth 后 INT 格式优于 FP 格式**。

- QK^T：INT8 CosSim 99.54% vs FP8 E4M3 92.83%
- 原因：smooth 后 Q/K 的分布趋于均匀，INT 的均匀量化点更合适；FP 的非均匀量化点反而浪费表示能力

这与 [[Rotation-based Quantization]] 中 [[A Comprehensive Evaluation on Quantization Techniques for LLMs|综合评测]] 的发现完全一致："旋转后 INT4 优于 FP4"。本质上 smoothing 和 rotation 都是让数据分布变均匀的预量化变换。

## 与其他压缩技术的关系

```
Attention 计算链:
  Q, K, V (存储) → Q·K^T (计算) → softmax → P·V (计算) → Output
       ↑                ↑                          ↑
  KV Cache 量化    Quantized Attention Kernel    同左
  (KIVI, KVQuant)  (SageAttention 系列)
  Token Eviction
  (H2O, SnapKV)
```

- **KV Cache 量化**（[[KV Cache Quantization]]）：压缩 K/V 的存储精度，减少内存
- **Quantized Attention Kernel**：压缩 attention 矩阵乘的计算精度，提升吞吐
- **Token Eviction**（[[Token Eviction]]）：减少参与计算的 token 数量
- 三者完全正交，可以叠加：KIVI 2-bit 存储 + SageAttention INT8 计算 + SnapKV 80% eviction

## 训练中的低比特 Attention

[[SageAttention3]] 的 SageBwd 是首个探索低比特 attention 训练的工作。关键发现：

1. **dO·V^T 是精度瓶颈**：反向传播 5 个 matmul 中，dO·V^T 的量化误差会沿序列长度方向累积到 dQ/dK。保持 dO·V^T 在 FP16，其余 4 个 matmul 用 INT8 即可
2. **Fine-tuning 无损**：Qwen2.5-3B 上 GSM8K 0.607 vs BF16 0.601，MMLU 0.653 vs 0.640
3. **Pretraining 收敛较慢**：低比特 attention 对 pretraining 的影响仍是开放问题
4. **INT8 训练 + FP4 推理有协同效应**：INT8 微调后用 FP4 推理，精度高于 BF16 微调 + FP4 推理

## 引用关系

← 理论基础：[[Emergent Outlier Features]]（Q/K outlier），[[Attention Sparsity]]（P 的稀疏分布），FlashAttention（tiling + online softmax）
← 技术关联：[[FP4 Quantization]]（NVFP4/MXFP4 格式），[[Rotation-based Quantization]]（smooth 后 INT 优于 FP）
← 已录入来源：[[SageAttention]]（ICLR 2025），[[SageAttention2]]（ICML 2025），[[SageAttention3]]（NeurIPS 2025）
→ 正交技术：[[KV Cache Quantization]]（存储量化），[[Token Eviction]]（cache 压缩），[[KV Cache Management]]
