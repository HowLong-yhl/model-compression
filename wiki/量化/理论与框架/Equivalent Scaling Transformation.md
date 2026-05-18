---
title: Equivalent Scaling Transformation
aliases: [等价缩放变换, Per-Channel Scaling, Smoothing]
tags: [model-compression, quantization, concept, technique]
created: 2026-04-20
updated: 2026-04-20
sources: [SmoothQuant, AWQ, A Comprehensive Evaluation on Quantization Techniques for LLMs, OSTQuant, FlatQuant]
---

## 定义

Equivalent Scaling Transformation 是一种在量化前对权重和/或激活施加 per-channel 缩放的技术，使得变换后的数学输出与原始完全一致，但量化误差分布更优。这是 [[SmoothQuant]] 和 [[AWQ]] 共享的核心技术基因。

## 基本原理

对线性层 `Y = XW`，插入对角缩放矩阵 `diag(s)`：

```
Y = X · W = (X · diag(s)⁻¹) · (diag(s) · W)
```

`diag(s)` 和 `diag(s)⁻¹` 互为逆矩阵，因此数学输出不变。但缩放改变了 X 和 W 的数值分布，从而影响量化误差。

## 两种应用方向

### SmoothQuant：平滑激活 → 实现 W8A8

**目标**：使激活的 per-channel 幅度变均匀（消除 [[Emergent Outlier Features]]），让 per-tensor INT8 量化变得可行。

```
s_j = max(|X_j|)^α / max(|W_j|)^{1-α}
```

- 大的 outlier 激活通道被除以大的 s_j → 变小
- 对应权重通道被乘以大的 s_j → 吸收难度
- `diag(s)⁻¹` 融合到前一层 LayerNorm 参数中 → 运行时零开销

### AWQ：保护 salient weights → 实现 W4A16

**目标**：缩放 salient weight channels 使其量化误差降低。

```
Q(w · s) · (x / s)
```

- Salient 权重通道乘以 `s > 1` → 相对量化误差降低
- 对应激活通道除以 `s` → 补偿
- 搜索空间：`s = s_X^α`，其中 `s_X` 为平均激活幅度

## 关键差异

| 维度 | [[SmoothQuant]] | [[AWQ]] |
|------|-----------------|---------|
| 缩放方向 | 缩小激活 outlier | 放大 salient weights |
| 量化目标 | W8A8 | W4A16 |
| α 含义 | 控制难度迁移比例 | 控制缩放强度 |
| 典型 α 范围 | 0.5-0.9 | 0-1（grid search） |
| 融合方式 | 融入 LayerNorm | 融入量化过程 |

## 为什么有效

等价缩放的有效性源于一个不对称性：**量化步长 Δ 由整个 group/channel 的 max 决定，而非单个值决定**。缩放单个通道会改变该通道的量化误差，但对整体 Δ 的影响是非线性的且通常较小——这创造了一个可以被优化利用的"间隙"。

## 与 Rotation 的关系

Scaling 和 [[Rotation-based Quantization|Rotation]] 是两种互补的预量化变换。根据 [[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 的系统评测：

- **有优化时**：Rotation + Scaling 组合效果最佳，Scaling 可微调旋转后的通道分布
- **无优化时**：单独 Rotation 优于 Rotation + calibrated Scaling（旋转后 max-based calibration 不准确）
- 最新方法（FlatQuant、OSTQuant）同时使用两者，并通过梯度优化联合学习

Scaling 也是更多方法的共享技术——OmniQuant、Outlier Suppression+、LQER 等都在不同阶段使用 Scaling。

## 优化 vs 未优化缩放

缩放方法可进一步细分为是否通过梯度学习：

- [[Unoptimized Scaling]]：公式/搜索确定（SmoothQuant α 公式, AWQ grid search）——快速、鲁棒、不过拟合
- [[Optimized Scaling]]：梯度优化可学习参数（OSTQuant learnable Λ, FlatQuant learnable diag(c)）——更精确、与旋转协同

核心 insight（来自 [[A Comprehensive Evaluation on Quantization Techniques for LLMs]]）：**旋转后不能用旧公式做 scaling**——旋转改变了分布，max-based 公式假设不再成立，必须联合优化。

## FP4 格式下的表现

在 [[FP4 Quantization]] 场景中，Scaling 仍然有效（因为它调整通道间平衡，与 group-wise FP4 正交），而 Rotation 对 MXFP4/NVFP4 几乎无效。这使得 Scaling 在 FP4 时代可能比 Rotation 更重要。
