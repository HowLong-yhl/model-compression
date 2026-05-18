---
title: Optimized Rotation
aliases: [优化旋转, Learned Rotation, Trainable Rotation]
tags: [model-compression, quantization, concept, technique, rotation, optimized]
created: 2026-04-20
updated: 2026-04-20
sources: [SpinQuant, OSTQuant, FlatQuant, A Comprehensive Evaluation on Quantization Techniques for LLMs]
---

## 定义

Optimized Rotation（优化旋转）指将旋转矩阵 O 作为**可学习参数**，通过梯度下降在 calibration 数据上端到端优化，使量化后的网络精度最大化。优化通常在 **Stiefel 流形**（所有正交矩阵的集合）上进行，以保证正交约束。

与 [[Unoptimized Rotation]]（随机/固定旋转）形成对比。

## 为什么需要优化旋转

[[SpinQuant]] 的关键发现揭示了优化的必要性：

- 100 次随机正交旋转在 W4A4 下的 zero-shot accuracy **差异高达 13 分**
- 即使都是 Hadamard 随机矩阵，方差仍有 **~6 分**
- 这意味着选到"坏"旋转可能导致严重性能退化

优化旋转解决了这一方差问题——Cayley SGD 优化后，方差降至 <1 分，且稳定超越所有随机试验中的最优者。

## 代表方法

| 方法 | 优化算法 | 旋转形式 | 正交约束 | 会议 |
|------|---------|---------|---------|------|
| [[SpinQuant]] | **Cayley SGD** | 全尺寸正交矩阵 R | 严格（Cayley Transform 保证） | ICLR 2025 |
| [[OSTQuant]] | **Riemann Adam** | 全尺寸正交矩阵 O | 严格（Stiefel 流形投影） | arXiv 2025 |
| [[FlatQuant]] | 标准 Adam | Kronecker 仿射 P₁⊗P₂ | **不严格约束**（一般可逆矩阵） | ICML 2025 |

## 三种流形优化方式对比

### 1. Cayley SGD（SpinQuant）

通过 Cayley Transform 将无约束梯度映射为正交更新：

```
R' = (I - α/2 · Y)⁻¹ · (I + α/2 · Y) · R
```

其中 Y 是反对称矩阵（由梯度投影得到）。Cayley Transform 保证 R' 始终正交。

- 优点：理论优雅，正交性精确保持
- 代价：需要矩阵逆运算（通过定点迭代高效实现，约 2× 普通 SGD 开销）

### 2. Riemann Adam（OSTQuant）

将 Adam 优化器推广到黎曼流形：

- 梯度通过正交投影映射到 Stiefel 流形的切空间
- 使用 retraction 操作将更新后的点拉回流形
- 支持自适应学习率（Adam 的优势）

- 优点：收敛更快（得益于 Adam 的动量和自适应性）
- 代价：实现复杂度略高

### 3. 无约束 + Kronecker 分解（FlatQuant）

不严格约束 P 为正交矩阵，而是学习一般可逆矩阵并通过 Kronecker 分解降低计算量：

```
P = P₁ ⊗ P₂,  P₁ ∈ R^{n₁×n₁}, P₂ ∈ R^{n₂×n₂}
```

- 优点：更灵活（可表示非正交变换），推理更快（Kronecker 乘法）
- 代价：理论保证弱于正交约束方法

## 优化目标与策略

所有方法共享类似的优化框架：

```
min_{O ∈ Stiefel} L(model_quantized(O) | data_calib)
```

| 方法 | 损失函数 | 校准数据量 | 优化迭代 | 计算时间 |
|------|---------|-----------|---------|---------|
| SpinQuant | Cross-entropy | 800 samples (WikiText2) | 100 iterations | 分钟级 |
| OSTQuant | KL-Top（top-k logits KL 散度） | 少量 | 块级优化 | **20 min** (A800) |
| FlatQuant | Block-wise MSE | 128 sentences | 逐层优化 | 小时级 |

## 优势

- **消除方差**：确定性地找到接近最优的旋转，不依赖运气
- **性能上限高**：在 W4A4 极端设定下显著优于随机旋转（SpinQuant 在 LLaMA-3 8B 上比 QuaRot 改善 45.1%）
- **可与缩放联合**：[[OSTQuant]] 和 [[FlatQuant]] 证明了联合优化 rotation + scaling 效果最佳

## 劣势

- **需要 calibration 数据**：无法做到 zero-shot（vs QuaRot 的零数据旋转）
- **计算开销**：需要梯度计算和多次前向传播
- **过拟合风险**：校准集与部署分布不匹配时可能退化（但实践中影响很小）
- **模型特异性**：每个模型需要独立优化旋转矩阵

## 理论支撑

### OSTQuant 的 QSUR 证明

QSUR（Quantization Space Utilization Rate）指标表明：
- 单独 Rotation 提升 QSUR（通过改变特征向量方向）
- **优化 Rotation 比随机 Rotation 提升更多**（因为优化可以对齐 eigenvector 方向与量化网格）
- Rotation + Scaling 组合达到**理论最大 QSUR**

### SpinQuant 的实验验证

优化旋转的效果不依赖初始化：从不同随机种子出发，Cayley SGD 均收敛到相似的高性能解（方差极小）。

## 与未优化旋转的对比

| 维度 | [[Unoptimized Rotation|未优化旋转]] | 优化旋转 |
|------|-----------|------|
| 数据依赖 | 无 | 需要 calibration 数据 |
| 计算开销 | 零 | 分钟~小时级 |
| 性能方差 | 大（6~13 分） | 极小（<1 分） |
| W4A4 表现 | 有方差，平均偏低 | 确定性高精度 |
| 部署形式 | 融入权重或在线 Hadamard | 同（融入权重或在线） |

## 适用场景

- **W4A4 / W4A4KV4 极端量化**：方差影响大，优化是必需的
- **小模型**（<8B）：小模型对量化更敏感，优化带来的收益更大
- **联合 W+A+KV 量化**：SpinQuant_had 配置，优化 R₁+R₂ 并用在线 Hadamard 做 R₃+R₄
- **与 [[Optimized Scaling]] 联合**：OSTQuant / FlatQuant 证明了联合优化的最优性

## 相关页面

- [[Rotation-based Quantization]]：旋转量化的总概览
- [[Unoptimized Rotation]]：随机/固定旋转方法
- [[Optimized Scaling]]：与优化旋转天然配对的缩放优化
