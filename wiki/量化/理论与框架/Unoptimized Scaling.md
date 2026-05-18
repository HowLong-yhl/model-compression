---
title: Unoptimized Scaling
aliases: [未优化缩放, Calibration-based Scaling, Formula-based Scaling]
tags: [model-compression, quantization, concept, technique, scaling, unoptimized]
created: 2026-04-20
updated: 2026-04-20
sources: [SmoothQuant, AWQ, A Comprehensive Evaluation on Quantization Techniques for LLMs]
---

## 定义

Unoptimized Scaling（未优化缩放）指通过**公式计算或简单搜索**确定缩放因子 s，**不使用反向传播（backpropagation）** 来优化。缩放因子基于激活/权重的统计量（如 max 值、均值）通过封闭公式或 grid search 得到。

与 [[Optimized Scaling]]（缩放因子作为可学习参数通过梯度下降优化）形成对比。

## 代表方法

| 方法              | 缩放因子计算方式                                    | 搜索空间            | 优化方式        | 会议         |
| --------------- | ------------------------------------------- | --------------- | ----------- | ---------- |
| [[SmoothQuant]] | `s_j = max(\|X_j\|)^α / max(\|W_j\|)^{1-α}` | α ∈ [0, 1]      | Grid search | ICML 2023  |
| [[AWQ]]         | `s = s_X^α`，s_X 为 per-channel 平均激活幅度        | α ∈ [0, 1]，20 步 | Grid search | MLSys 2024 |

## 工作原理

### SmoothQuant 的公式缩放

核心思路：观察到激活有 outlier 而权重均匀，通过 per-channel 缩放将激活难度"迁移"给权重。

```
Y = X · W = (X · diag(s)⁻¹) · (diag(s) · W)
```

缩放因子由**显式公式**计算：
```
s_j = max(|X_j|)^α / max(|W_j|)^{1-α}
```

- α 控制难度迁移程度：α=0.5 为均衡分配，α=0.9 将大部分难度推给权重
- α 通过在 calibration 子集上的**快速 grid search** 选定
- 整个过程**无需梯度计算**

### AWQ 的激活感知缩放

核心思路：放大 salient weight channels 的相对幅度，降低其量化误差。

```
Q(w · s) · (x / s) ≈ w · x
```

搜索过程：
- 搜索空间：`s = s_X^α`，其中 s_X 为每个通道的平均激活幅度
- α ∈ [0, 1]，均匀分 20 步
- 对每个 α 值计算 MSE，选最小的
- **搜索开销极低**——仅需 20 步前向评估（GPTQ 需收集全部 calibration 激活并逐层计算 Hessian）

## 核心特征："无反向传播"

未优化缩放的本质定义特征是 **不使用梯度信息**：

1. **SmoothQuant**：缩放因子完全由统计量的封闭公式决定
2. **AWQ**：缩放因子通过前向 MSE 评估选择最优 α，但不计算梯度

这意味着：
- 计算开销极低（仅需少量前向传播）
- 不会过拟合 calibration 数据
- 对 calibration 分布偏移鲁棒（AWQ 仅需 16 samples）

## 优势

- **极快**：SmoothQuant 几乎瞬时完成；AWQ 仅需 20 次前向评估
- **鲁棒性强**：AWQ 在 domain shift 下 PPL 退化仅 +0.5~0.6（GPTQ 退化 +2.3~4.9）
- **不过拟合**：无梯度 = 无 calibration 过拟合
- **极少数据**：AWQ 仅需 **16 sentences**，SmoothQuant 用 512 samples 仅做统计收集
- **零推理开销**：缩放融入 LayerNorm 或权重，部署时完全消失

## 劣势

- **非最优**：公式/搜索只能覆盖有限空间，无法找到全局最优
- **启发式**：SmoothQuant 的 α 公式和 AWQ 的 `s=s_X^α` 参数化都是经验设计
- **W4A4 下效果有限**：[[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 实验发现，在无优化旋转之后，calibrated scaling 反而可能**损害性能**（因为旋转后 max-based calibration 不准确）
- **通道级粒度**：只能做 per-channel 调整，无法改变通道间的"方向"（这是 rotation 的优势）

## 与综述论文的关键发现

[[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 的实验结论：

> **无优化时**：单独随机旋转 **优于** 旋转 + calibrated scaling

原因分析：旋转后 outlier 已被均匀化，分布变得近似均匀高斯。此时基于 max 值的 calibration scaling 公式不再准确——它假设存在 outlier channel 需要被平衡，但旋转后这个假设不成立。

这解释了为什么需要从 [[Unoptimized Scaling]] 升级到 [[Optimized Scaling]]——在旋转后的新分布上，需要端到端优化才能找到正确的缩放。

## 与优化缩放的对比

| 维度 | 未优化缩放 | [[Optimized Scaling|优化缩放]] |
|------|-----------|------|
| 优化方式 | 公式 / Grid search | 梯度下降 |
| 数据需求 | 极少（16~512 samples） | 较多（128+ sentences） |
| 计算开销 | 极低（秒级） | 较高（分钟级） |
| 过拟合风险 | 无 | 存在（但可控） |
| W8A8 表现 | 优秀（SmoothQuant 近无损） | 无必要（已够好） |
| W4A4 表现 | 有限（尤其旋转后） | 显著更好 |
| 代表方法 | [[SmoothQuant]], [[AWQ]] | [[OSTQuant]], [[FlatQuant]], OmniQuant |

## 适用场景

- **W8A8 部署**：SmoothQuant 的最佳战场，近乎无损
- **W4A16 weight-only**：AWQ 的最佳战场，极少 calibration 即可
- **快速原型**：无需长时间优化，秒级完成量化
- **大模型（70B+）**：数据鲁棒性和低开销尤为重要
- **多模态/新领域模型**：AWQ 的 domain-shift 鲁棒性是关键优势

不适用场景：W4A4 极端量化 + 与 rotation 组合（此时应使用 [[Optimized Scaling]]）。

## 相关页面

- [[Equivalent Scaling Transformation]]：缩放变换的总概览
- [[Optimized Scaling]]：通过梯度学习的缩放方法
- [[Unoptimized Rotation]]：与未优化缩放同为"无梯度"的预量化技术
