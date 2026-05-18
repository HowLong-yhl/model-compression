---
title: Unoptimized Rotation
aliases: [未优化旋转, Random Rotation, Fixed Rotation]
tags: [model-compression, quantization, concept, technique, rotation, unoptimized]
created: 2026-04-20
updated: 2026-04-20
sources: [QuIP#, QuaRot, A Comprehensive Evaluation on Quantization Techniques for LLMs]
---

## 定义

Unoptimized Rotation（未优化旋转）指在量化前使用**随机生成或固定的正交矩阵**对权重/激活进行旋转变换，不通过梯度下降学习旋转矩阵本身。其效果依赖于正交变换的统计性质（如 incoherence），而非数据驱动的优化。

与 [[Optimized Rotation]]（通过梯度学习旋转矩阵）形成对比。

## 代表方法

| 方法 | 旋转矩阵来源 | 推理方式 | 会议 |
|------|-------------|---------|------|
| QuIP | 随机正交矩阵（Kronecker 积） | 在线变换 | NeurIPS 2023 |
| [[QuIP#]] | **随机 Hadamard + 符号向量** (RHT) | 在线 Hadamard | ICML 2024 |
| [[QuaRot]] | **随机 Hadamard 矩阵** | 融入权重 + 在线 Hadamard | NeurIPS 2024 |

## 工作原理

### 为什么随机旋转能消除 outlier

正交矩阵的几何意义是刚性旋转（保距变换）。根据中心极限定理，当大量通道被线性混合后，结果趋向高斯分布——outlier 被"稀释"到所有维度中。

具体地，对 n 维向量 x，应用 Hadamard 矩阵 H 后，每个输出维度是输入所有维度的等权求和（仅改变符号）：

```
(Hx)_i = (1/√n) · Σⱼ (±1) · x_j
```

即使 x 中有单个极大值，旋转后该值被分散到 n 个维度中，每个维度仅增加 O(1/√n)。

### Hadamard 矩阵的特殊优势

绝大多数未优化旋转方法选择 Hadamard 矩阵而非一般正交矩阵：

- **理论优势**：[[QuIP#]] 证明了 Hadamard 旋转的 incoherence bound 为 O(log n)，优于 Kronecker 正交矩阵的 O(log² n)
- **计算优势**：Fast Walsh-Hadamard Transform 仅需 O(n log n) 操作，且全为加减法（无浮点乘法）
- **实践验证**：[[SpinQuant]] 实验显示 Hadamard 随机矩阵整体优于一般随机正交矩阵

### QuaRot 的零数据方法

[[QuaRot]] 展示了未优化旋转最极致的应用——旋转步骤**完全不需要 calibration 数据**：

1. 利用 Computational Invariance 定理，Hadamard 矩阵可直接融入权重
2. 仅需将 RMSNorm 的 scale 参数 α 先融入相邻权重
3. 整个 Stage 1 是纯代数操作，与数据无关

## 优势

- **零校准数据**（QuaRot 的旋转步骤）：纯代数操作，无过拟合风险
- **理论保证**：incoherence theory 给出了误差上界（[[QuIP#]] Theorem 4.1）
- **极快部署**：无需优化迭代，秒级完成
- **通用性**：对任何 pre-norm Transformer 都适用

## 劣势

- **性能方差大**：[[SpinQuant]] 的关键发现——100 次随机旋转的 W4A4 accuracy 差异高达 **13 分**，即使 Hadamard 随机矩阵仍有 ~6 分方差
- **非最优**：对于特定模型和量化设定，存在比任何随机选择都更好的旋转矩阵
- **无法适配数据**：不利用 calibration 数据的信息，可能在某些层效果不佳

## 关键实验数据

来自 [[SpinQuant]] 的方差实验（LLaMA-2 7B, W4A4）：

| 旋转类型 | 最佳 accuracy | 最差 accuracy | 方差 |
|---------|-------------|-------------|------|
| 随机正交矩阵 | ~55% | ~42% | **13 分** |
| 随机 Hadamard | ~57% | ~51% | **6 分** |
| Cayley 优化（参考） | ~58% | ~57% | <1 分 |

## 与优化旋转的对比

| 维度 | 未优化旋转 | [[Optimized Rotation|优化旋转]] |
|------|-----------|------|
| 数据依赖 | 无（或仅用于后续 GPTQ 步骤） | 需要 calibration 数据 |
| 计算开销 | 零（秒级） | 需要优化迭代（分钟~小时） |
| 性能方差 | 大（6~13 分） | 极小（<1 分） |
| 最优性 | 统计最优（平均好） | 确定性最优 |
| 代表方法 | QuIP#, QuaRot | [[SpinQuant]], [[FlatQuant]], [[OSTQuant]] |

## 适用场景

- **大模型 + 高比特**（W4A8, W6A6）：随机旋转已足够（QuaRot 在 70B+6bit 下无损）
- **快速部署**：不需要 calibration 流程
- **Weight-only 极压**（2-3 bit）：配合 E₈ lattice VQ（[[QuIP#]] 路线）效果已很强
- **资源受限**：无需 GPU 做优化

不适用场景：W4A4 极端设定 + 小模型（此时方差影响大，应转向 [[Optimized Rotation]]）。

## 相关页面

- [[Rotation-based Quantization]]：旋转量化的总概览
- [[Optimized Rotation]]：通过梯度学习的旋转方法
- [[Lattice Codebooks]]：与未优化旋转天然配合的 VQ 码本（QuIP# 路线）
- [[Quantization Insights|Insight 2]]：为什么未优化旋转后不应加 calibrated scaling
