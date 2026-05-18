---
title: Optimized Scaling
aliases: [优化缩放, Learned Scaling, Trainable Scaling]
tags: [model-compression, quantization, concept, technique, scaling, optimized]
created: 2026-04-20
updated: 2026-04-20
sources: [OSTQuant, FlatQuant, A Comprehensive Evaluation on Quantization Techniques for LLMs]
---

## 定义

Optimized Scaling（优化缩放）指将缩放因子 S 作为**可学习参数**，通过梯度下降端到端优化，使量化后的网络精度最大化。与 [[Unoptimized Scaling]]（公式/搜索确定）的核心区别是：缩放因子不再由 `s = max(|X|)^α / max(|W|)^{1-α}` 这样的公式计算，而是在损失函数上做**反向传播**得到。

与 [[Optimized Rotation]] 天然配对——[[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 和 [[OSTQuant]] 均证明了两者联合优化效果最佳。

## 代表方法

| 方法 | 缩放参数化 | 联合优化旋转 | 优化方式 | 会议 |
|------|-----------|------------|---------|------|
| OmniQuant | learnable diag(s) + shift z | 否 | Block-wise 梯度 | ICLR 2024 |
| [[FlatQuant]] | learnable diag(c) + clipping α | **是**（Kronecker P） | Block-wise MSE 梯度 | ICML 2025 |
| [[OSTQuant]] | learnable diag(Λ) | **是**（正交 O） | Riemann Adam + Adam | arXiv 2025 |

## 为什么需要优化缩放

### 未优化缩放的局限

[[SmoothQuant]] 和 [[AWQ]] 的缩放公式基于两个隐含假设：
1. 激活中存在明显的 per-channel outlier
2. max 值是通道"难度"的良好代理

但这两个假设在以下情况下失效：
- **旋转后**：rotation 使分布变均匀，max-based 公式不再适用
- **W4A4 极端精度**：4-bit 量化的非线性使得简单公式无法捕捉最优平衡点
- **通道间交互**：公式方法独立处理每个通道，忽略了通道间的量化误差传播

### 优化缩放的理论必要性

[[OSTQuant]] 通过 QSUR 理论证明：

```
QSUR_max ∝ T = c · Λ^{-1/2} · Q⊤
```

其中 Λ^{-1/2} 对应**缩放**（消除特征值不均），Q⊤ 对应**旋转**（消除 outlier 方向）。两者缺一不可——**单独 rotation 无法达到最大 QSUR**。

但要达到 Λ^{-1/2} 这个理论最优缩放，需要知道精确的协方差矩阵特征值——这不是简单公式能给出的，需要数据驱动的优化。

## 方法对比

### OmniQuant（早期探索者）

- 将 SmoothQuant 的 s 和 shifting z 都变为可学习参数
- 使用 block-wise 损失函数训练
- **仅优化缩放**，不做旋转优化
- 效果：在 [[SmoothQuant]] 基础上提升明显，但仍弱于加入旋转的方法

### FlatQuant（ICML 2025）

联合优化三组参数：
```
Θ = {P₁⊗P₂ (旋转), diag(c) (缩放), α_w/α_a (clipping)}
```

- diag(c) 实现 per-channel 缩放，可融入前一层 LayerNorm（零推理开销）
- 缩放的目的：在仿射变换之前先粗调通道平衡
- 与 P 矩阵互补：c 处理 scale 差异，P 处理方向性 outlier

### OSTQuant（arXiv 2025）

变换对定义为 `T = Λ · O`：
```
y = Q(x · W₁ · O · Λ) · Q(Λ⁻¹ · O⊤ · W₂)
```

- Λ 为对角正定矩阵，直接学习对角元素
- O 为正交矩阵，用 Riemann Adam 优化
- 初始化：Λ = I（单位矩阵），O 由权重协方差特征分解确定（WOMI）
- 部署时 `Λ⁻¹ · O⊤ · W₂` 预计算融入权重，**零推理开销**

## 优势

- **理论最优**：QSUR 证明 optimized scaling + rotation 达到最大量化空间利用率
- **超越公式上限**：不受 `s=max(|X|)^α` 参数化的限制
- **与旋转协同**：旋转后的新分布需要匹配的缩放，优化能自动找到
- **零推理开销**：所有可学习缩放都可融入权重或 LayerNorm
- **适配 W4A4**：在极端量化设定下优势最大

## 劣势

- **需要 calibration 数据 + 梯度计算**：比 [[Unoptimized Scaling]] 开销大
- **可能过拟合**：OSTQuant 通过 KL-Top loss 缓解，FlatQuant 通过 block-wise MSE 限制
- **不如未优化缩放鲁棒**：AWQ 的跨域鲁棒性优势在优化方法中不保证
- **更复杂的部署流程**：需要额外的 calibration 步骤

## 关键实验证据

来自 [[A Comprehensive Evaluation on Quantization Techniques for LLMs]]：

| 配置 | 方法 | PPL |
|------|------|-----|
| Rotation only | QuaRot (random Had) | baseline |
| Rotation + unoptimized scaling | QuaRot + calibrated s | **更差**（!） |
| Rotation + **optimized scaling** | optimized R + optimized S | **最佳** |

关键insight：**旋转后不能用旧公式做 scaling——必须联合优化**。

## 2×2 分类矩阵总览

```
              │  Unoptimized (公式/搜索)  │  Optimized (梯度学习)
─────────────┼──────────────────────────┼─────────────────────────
  Rotation   │  QuIP#, QuaRot           │  SpinQuant, OSTQuant, FlatQuant
  (旋转)     │  随机 Hadamard / RHT     │  Cayley SGD / Riemann Adam
─────────────┼──────────────────────────┼─────────────────────────
  Scaling    │  SmoothQuant, AWQ        │  OSTQuant, FlatQuant, OmniQuant
  (缩放)     │  α 公式 / grid search    │  learnable diag(s), 梯度优化
```

最佳组合（综述论文结论）：**Optimized Rotation + Optimized Scaling + GPTQ + Low-rank**

## 适用场景

- **W4A4 / W4A4KV4 极端量化**：必须联合优化才能获得可用精度
- **与 [[Optimized Rotation]] 配合**：两者联合效果远超单独使用
- **精度敏感应用**：需要榨干量化模型的最后一点性能

不适用场景：W8A8（SmoothQuant 已够好）、快速部署（AWQ 更快）、跨域鲁棒性要求高（AWQ 更稳）。

## 相关页面

- [[Equivalent Scaling Transformation]]：缩放变换的总概览
- [[Unoptimized Scaling]]：公式/搜索确定的缩放方法
- [[Optimized Rotation]]：与优化缩放天然配对的旋转优化
