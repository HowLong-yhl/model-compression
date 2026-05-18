---
title: Hadamard Transform
aliases: [Fast Walsh-Hadamard Transform, FWHT, Hadamard Matrix, RHT, 哈达玛变换]
tags: [model-compression, quantization, concept, technique, algorithm]
created: 2026-04-20
updated: 2026-04-20
sources: [QuIP#, QuaRot, SpinQuant, FlatQuant, OSTQuant]
---

## 定义

Hadamard Transform 是旋转量化（[[Rotation-based Quantization]]）中最常用的正交变换实现方式。Hadamard 矩阵 H 是一种特殊的正交矩阵，其元素仅为 ±1/√n，这使其既能实现正交变换的 outlier 消除效果，又避免了通用矩阵乘法的计算瓶颈。

## 数学定义

### Hadamard 矩阵的递归构造

n×n 的 Hadamard 矩阵通过 Sylvester 构造递归定义（n 必须为 2 的幂）：

```
H_1 = [1]

H_2 = (1/√2) * [ 1  1 ]
                [ 1 -1 ]

H_{2n} = (1/√2) * [ H_n   H_n  ]
                   [ H_n  -H_n  ]
```

关键性质：
- **正交性**：H·H^T = I（Hadamard 是正交矩阵）
- **对称性**：H = H^T（故 H^{-1} = H）
- **元素约束**：所有元素为 ±1/√n，无浮点乘法需求
- **保距性**：不改变 Frobenius 范数，只重新分配能量


## 变换前后按通道mix分布差异大
原activation分布：按channel分布，有outlier channel
![[images/Pasted image 20260424114736.png]]
hadamard transform后分布：按channel mix，channel峰值不明显
![[images/Pasted image 20260424114915.png]]

### Randomized Hadamard Transform (RHT)

[[QuIP#]] 使用的增强版本：先乘随机对角符号矩阵 D（diag(±1)），再乘 Hadamard：

```
RHT(X) = X · D · H
```

D 的随机符号使变换具有更强的 incoherence 保证——理论 bound 为 O(log(mn))，优于纯 Hadamard 的确定性 bound。

## 计算复杂度

### Fast Walsh-Hadamard Transform (FWHT)

类似 FFT 的分治算法，利用 Hadamard 矩阵的递归结构：

| 操作 | 复杂度 | 备注 |
|------|--------|------|
| 通用矩阵乘法 | O(n²) | 对 n×n 正交矩阵 |
| **FWHT** | **O(n log n)** | 利用递归结构 |
| 实际开销 | **~8% 推理延迟增加** | SpinQuant 在线 Hadamard 实测 |

FWHT 的核心优势：
- 无需存储完整矩阵（由递归公式隐式定义）
- 所有操作为加减法（±1/√n 乘法可合并到 scaling）
- 适合 GPU 并行（蝶形结构天然可并行）

### 维度兼容性

Hadamard 矩阵要求维度为 2 的幂。实际 LLM 中处理方式：
- 大多数主流模型的隐藏维度已是 2 的幂（4096, 8192 等）
- 不兼容时可 padding 到最近 2 的幂，或使用 Kronecker 乘积的小 Hadamard 块

## 在量化中的应用

### 为什么选择 Hadamard 而非通用正交矩阵

1. **计算效率**：O(n log n) vs O(n²)，使在线旋转变得可行
2. **无需训练/存储**：矩阵由公式定义，零参数
3. **Incoherence 保证**：RHT 理论上保证变换后分布的 incoherence bound 为 O(log n)
4. **硬件友好**：仅需加减法，适配 INT/FP 硬件

### 各方法中的使用方式

| 方法 | Hadamard 用法 | 是否在线计算 | 能否融入权重 |
|------|--------------|-------------|-------------|
| [[QuIP#]] | RHT（D·H）作为 incoherence preprocessing | 离线 | 是（融入权重） |
| [[QuaRot]] | 随机 Hadamard 在 4 个位置 | 部分在线 | 大部分可融入 |
| [[SpinQuant]] | Hadamard 作为旋转初始化 → Cayley SGD 优化 | 部分在线 | 大部分可融入 |
| [[OSTQuant]] | 作为对比 baseline（优化后超越） | — | — |
| [[FlatQuant]] | 不使用（改用 Kronecker 可学习仿射） | — | — |

### 旋转位置

在 Transformer 中，旋转通常插入在 4 个位置（以 SpinQuant/QuaRot 的符号）：

```
R₁: 残差流中（RMSNorm 前后）     ← 可融入权重，零开销
R₂: FFN 中间层                    ← 可融入权重，零开销
R₃: K/V 投影后（post-RoPE）       ← 需在线计算（与 KV-cache 交互）
R₄: V 投影后（Attention 输出前）  ← 需在线计算
```

R₁ 和 R₂ 通过 Computational Invariance 定理（来自 SliceGPT）可完全融入相邻权重矩阵，部署时完全消失。R₃ 和 R₄ 必须在推理时在线计算——这是 ~8% 延迟开销的来源。

## 与 Incoherence 理论的关系

[[QuIP#]] 证明了一个深刻联系：各种 outlier suppression heuristic（scaling、mixed-precision 等）本质上都在追求 **[[Incoherence]]**（即消除矩阵中的"尖峰"元素）。Hadamard/RHT 是实现 incoherence 的数学上最优雅且有理论保证的方式。

具体来说：如果权重矩阵 W 经 RHT 处理后变为 µ-incoherent，则：

```
max|W̃_ij| ≤ µ · ‖W‖_F / √(mn)
```

µ 越小，说明 outlier 被消除得越彻底，量化就越容易。

## 局限性

1. **对 FP4 格式几乎无效**（[[Quantization Insights|Insight 7]]）：MXFP4/NVFP4 的小 group size（16-32）本身就隔离了 outlier 的影响，Hadamard 旋转的全局分散没有边际收益
2. **引入方差**（SpinQuant 发现）：随机 Hadamard（含 RHT）在不同随机种子下 accuracy 差异可达 13 分
3. **维度限制**：严格要求 2 的幂维度，实际中大模型基本满足但小模型可能需要 padding
4. **在线部分有延迟开销**：R₃/R₄ 的在线 Hadamard 增加约 8% 推理延迟

## 相关页面

- [[Rotation-based Quantization]]：旋转量化的总概览
- [[Incoherence]]：Hadamard/RHT 所追求的理论目标，量化的本质
- [[Unoptimized Rotation]]：使用固定/随机 Hadamard 的方法（QuIP#, QuaRot）
- [[Optimized Rotation]]：以 Hadamard 为初始化再优化的方法（SpinQuant, OSTQuant）
- [[Quantization Insights]]：Hadamard 对 INT4 有效但对 FP4 无效的分析（Insight 7）
