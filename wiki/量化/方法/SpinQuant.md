---
title: "SpinQuant: LLM Quantization with Learned Rotations"
aliases: [SpinQuant]
tags: [model-compression, quantization, source, rotation, PTQ, learned-rotation]
created: 2026-04-20
updated: 2026-04-20
type: source-summary
venue: ICLR 2025
authors: [Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, Tijmen Blankevoort]
affiliation: Meta
url: https://arxiv.org/abs/2405.16406
code: https://github.com/facebookresearch/SpinQuant
---

## 一句话总结

SpinQuant 发现随机旋转矩阵在量化中引入巨大性能方差（最多 13 分差距），因此提出在 **Stiefel 流形上用 Cayley SGD 学习最优旋转矩阵**，在 W4A4KV4 设定下显著优于所有随机旋转方法。

## 核心问题

虽然[[Rotation-based Quantization|旋转量化]]能消除 outlier，但不同随机旋转矩阵会导致量化网络在 zero-shot 推理任务上**相差高达 13 个百分点**。如何找到最优旋转？

## 方法详解

### 旋转参数化（四个旋转矩阵）

SpinQuant 识别了 Transformer 中四个可插入旋转而不影响全精度输出的位置：

| 旋转 | 位置 | 是否可融入权重 | 作用 |
|------|------|---------------|------|
| R₁ | Residual stream（Embedding 后） | ✅ 可融入相邻权重 | 消除残差流中的 outlier |
| R₂ | V → O_proj（head-wise，维度 D_head × D_head） | ✅ 可融入 W_v 和 W_o | 消除注意力值投影中的 outlier |
| R₃ | Q/K 在 RoPE 之后 | ❌ 需在线计算 | 消除 KV cache 中的 outlier |
| R₄ | FFN down projection 输入之前 | ❌ 需在线计算 | 消除 MLP 中间激活的 outlier |

- **SpinQuant_no_had**：仅使用 R₁ + R₂（可完全融入权重，零推理开销）
- **SpinQuant_had**：使用全部 R₁~R₄（R₃/R₄ 用 Hadamard 在线计算，适用于 W4A4KV4）

### Cayley SGD 优化

旋转矩阵 R 必须保持正交约束（R^T R = I），即优化空间是 **Stiefel 流形 M**。SpinQuant 使用 Cayley Transform 来保证每步更新后的矩阵仍正交：

```
R' = (I - α/2 · Y)⁻¹ (I + α/2 · Y) · R
```

其中 Y 是由梯度投影得到的反对称矩阵：`Y = Ĝ - Ĝ^T`，`Ĝ = G·R^T - 1/2·R·R^T·G·R^T`

**优化目标**：最小化量化网络在校准集上的任务损失（cross-entropy），固定所有原始权重 W，仅优化 {R₁, R₂}：

```
argmin_{R∈M} L_Q(R₁, R₂ | W, X)
```

- 旋转参数仅占原始权重的 ~0.26%
- 使用 WikiText2 的 800 样本做校准
- 仅需 100 次迭代即可收敛

### 关键 insight

1. Hadamard 随机矩阵整体优于一般随机正交矩阵（与 [[QuIP#]] 的理论一致：Hadamard 有更紧的 max-entry bound）
2. 但即使 Hadamard 随机矩阵仍有 ~6 分方差
3. Cayley 优化后方差极小，且稳定优于最好的随机旋转

## 实验亮点

| 设定 | 模型 | SpinQuant vs QuaRot | SpinQuant vs Full-precision |
|------|------|-------|------|
| W4A4KV4 | LLaMA-2 7B | +2.9↑ | 仅 -2.9 gap |
| W4A4KV4 | LLaMA-3 8B | **相对改善 45.1%** | — |
| W4A8KV8 | Mistral 7B | 12.1 → 1.6 gap | 仅 -1.6 gap |

- SpinQuant_no_had 的 W4A8 性能可媲美 QuIP# / OmniQuant 等纯 weight-only 方法
- 优于 LLM-QAT 19.1 分（W4A4KV4），优于 SmoothQuant 25.0 分

## 与其他方法的关系

- **vs [[QuIP#]]**：QuIP# 也使用 Hadamard 旋转，但只做随机旋转+lattice codebook；SpinQuant 优化旋转本身
- **vs QuaRot**：QuaRot 使用随机 Hadamard；SpinQuant 学习旋转，在 LLaMA-3 8B 上改善 45.1%
- **vs [[SmoothQuant]] / [[AWQ]]**：Scaling 方法在 W4A4 下效果有限；rotation 是 W4A4 的关键推动力
- **vs FlatQuant / OSTQuant**：后续工作进一步结合 learned rotation + scaling

## 局限性

- 需要校准数据和优化过程（虽然仅 100 iterations，但比纯随机方法开销大）
- 在线 Hadamard 旋转（R₃/R₄）引入约 8% 推理延迟
- 论文未探索 2-bit 量化

## 引用关系

← 被启发：[[Rotation-based Quantization]] (QuaRot/SliceGPT), Cayley SGD (Li et al. 2020)
← 分类：[[Optimized Rotation]]（Cayley SGD 学习正交旋转）
← 对比：[[Unoptimized Rotation]]（随机 Hadamard 方差可达 13 分差距）
→ 启发了：FlatQuant, OSTQuant 等同时优化旋转+缩放的方法
→ KV cache 量化：SpinQuant 的 R₃ 旋转专门针对 KV cache 中的 outlier，W4A4**KV4** 方案是全量化框架下的 KV cache 量化。后续专门的 [[KV Cache Quantization]] 方法（如 RotateKV）将旋转策略从全局固定升级为 outlier-aware adaptive
→ 相关 Insight：[[Quantization Insights]]（Insight 1-2 的核心证据）
