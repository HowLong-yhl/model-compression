---
title: Incoherence
aliases: [Incoherence Processing, µ-incoherent, 不相干性]
tags: [model-compression, quantization, concept, theory, rotation]
created: 2026-04-23
updated: 2026-04-23
sources: [QuIP#, QuaRot, SpinQuant]
---

## 定义

Incoherence 是衡量矩阵中**能量分布均匀程度**的数学指标，直接决定了矩阵的量化难度。一个 m×n 矩阵 W 被称为 µ-incoherent，当且仅当：

```
max|W_ij| ≤ µ · ‖W‖_F / √(mn)
```

右边的 `‖W‖_F / √(mn)` 就是矩阵所有元素的 RMS（均方根），可以理解为"平均幅度"。因此这个不等式的含义是：**矩阵中最大的元素不能超过平均幅度的 µ 倍**。

- µ 越小 → 越 incoherent → 能量越均匀分布 → **越容易量化**
- µ 越大 → 越 coherent → 存在 outlier "尖峰" → **量化灾难**

理想状态 µ = 1 意味着所有元素幅度完全相同——此时 per-tensor 量化的信息利用率最高。

## 为什么 Incoherence 是量化的本质

量化误差的根源在于：量化范围（scale factor）由 `max|X_ij|` 决定，而量化精度由 bit 数决定。当少数 outlier 元素的幅度远超平均值（µ 很大）时：

```
量化步长 s = max|X_ij| / (2^{b-1} - 1)
```

这个步长对 99% 的正常值来说太粗了——大部分正常值被压缩到仅 2-3 个量化级中，有效精度远低于 b bit。

用 incoherence 的语言来说：如果 µ = 100（outlier 是均值的 100 倍），那么 4-bit 的 16 个量化级中有 ~14 个被浪费在空旷的 outlier 区域。只有当 µ ≈ 1 时，所有量化级才被充分利用。

这就是 [[QuIP#]] 的核心论点：**所有 outlier suppression 技术——scaling（[[SmoothQuant]]、[[AWQ]]）、mixed-precision（[[LLM.int8()]]）、rotation（[[QuaRot]]、[[SpinQuant]]）——本质上都在追求 incoherence，只是手段和理论保证不同。**

## 各方法的 Incoherence 处理策略

| 方法 | 策略 | 理论保证 | µ 改善程度 |
|------|------|---------|-----------|
| 原始矩阵（无处理） | — | — | µ 可能极大（outlier 幅度 20-100× 均值） |
| [[SmoothQuant]] / [[AWQ]] | 通道级缩放 | 无理论 bound | 经验有效，但不保证彻底消除 |
| [[LLM.int8()]] | 混合精度分解 | 无理论 bound | outlier 通道用 FP16 绕过，不直接降低 µ |
| QuIP（Kronecker 正交） | Kronecker 随机正交 × 2 | µ = **O(log²(mn))** | 理论保证，但 bound 较松 |
| [[QuIP#]] / [[QuaRot]] (RHT) | 随机 Hadamard 变换 | µ = **O(√log(mn))** | 理论最优，bound 最紧 |
| [[SpinQuant]] | 学习最优旋转 | 无显式 bound，但经验上最优 | 消除随机旋转的方差 |

### RHT 的 Incoherence Bound 推导直觉

随机 Hadamard 变换 `X̂ = X · D · H`（D 为随机 ±1 对角矩阵，H 为 Hadamard 矩阵）为什么能保证 incoherence？

**几何直觉**：Hadamard 变换把每个新 channel 定义为所有原始 channel 的等权重 ±1/√n 线性组合。假设原始 channel k 有个幅度为 A 的 outlier，变换后它的能量被均匀分摊到 n 个新 channel 上，每个新 channel 仅收到 A/√n 的贡献。

**随机符号 D 的作用**：纯 Hadamard 矩阵是确定性的，对某些特殊结构的矩阵可能失效（outlier 恰好与 Hadamard 行对齐）。随机符号翻转打破了这种对齐，使 bound 以高概率成立。这也是 [[SpinQuant]] 发现不同随机种子导致 13 分方差的根源——D 的选择影响了具体的 incoherence 质量。

**概率论论证**：变换后的每个元素 X̂_ij 是 n 个独立随机变量 ±X_ik/√n 的和。由 Hoeffding 不等式，`|X̂_ij|` 以高概率不超过 O(√(log(n)/n)) · ‖X_i‖_2。对所有 mn 个元素取 union bound，得到全局 incoherence µ = O(√log(mn))。

### Kronecker vs RHT：Bound 的差异

| 方法 | Incoherence bound (µ) | 计算复杂度 | 说明 |
|------|----------------------|-----------|------|
| QuIP (Kronecker) | O(log²(mn)) | O(n√n) | 两个独立随机正交矩阵的 Kronecker 积 |
| QuIP# / QuaRot (RHT) | **O(√log(mn))** | **O(n log n)** | 随机符号 + Hadamard |

RHT 同时在理论（更紧的 bound）和计算（更低的复杂度）上双赢。这是 QuIP# 相对 QuIP 的核心改进之一。

## 可视化理解

以 LLaMA 第 12 层 self_attn.k_proj 的激活为例：

**变换前**（高 coherence / µ 大）：大部分 channel 的 25th-75th percentile 在 ±1 以内，但少数 outlier channel 的 Min/Max 冲到 ±10~15。这些 outlier 决定了量化范围，导致正常 channel 的有效量化精度极低。

**Hadamard 变换后**（低 coherence / µ 小）：所有 channel 的 Min/Max 包络线变得高度均匀（约 ±5），不再有明显的尖峰。总能量不变（Frobenius 范数保持），但从"少数 channel 极大、多数 channel 极小"变为"所有 channel 幅度接近"。此时 per-tensor 量化的每个量化级都被充分利用。

这正是 incoherence 改善的直接视觉证据：µ 从远超 1 被压缩到接近理论下界。

## 与 Lattice Codebook 的深层联系

[[QuIP#]] 揭示了 incoherence 与向量量化之间的深层逻辑链：

```
高 incoherence (outlier) → 分布高度非球形 → 标量量化勉强可用，VQ 无优势
       ↓ RHT
低 incoherence (均匀)   → 分布近似球形高斯 → 球形 VQ 码本（E₈ 格）最优
```

经 RHT 处理后，权重分布变为近似球形高斯。此时：
- **标量量化**（SQ）用超立方体覆盖球形分布，覆盖效率低（角落浪费空间）
- **E₈ 格码本**是 8 维球堆积的理论最优解（kissing number = 240），覆盖效率最高

因此 incoherence processing 不仅让简单量化变得可行，还为更高级的向量量化创造了理想条件。详见 [[Lattice Codebooks]]。

## Incoherence 的局限

1. **理论 bound vs 实际表现有 gap**：O(√log(mn)) 是 worst-case bound，实际中 RHT 通常远优于此；但不同随机种子之间仍有差异（SpinQuant 的 ~6 分方差），说明理论 bound 虽紧但不够——最优旋转仍需学习
2. **对 FP4 格式效果有限**（[[Quantization Insights|Insight 7]]）：MXFP4/NVFP4 的极小 group size（16-32）本身就在局部实现了 incoherence，全局 Hadamard 旋转的边际收益很小
3. **仅处理幅度不均，不处理分布形状**：incoherence 让所有 channel 幅度接近，但如果底层分布本身不适合均匀量化网格（如重尾分布），仍需配合 scaling 或非均匀格式
4. **仅是量化误差的"一半"**：[[Concentration-Alignment]] 证明 SQNR 分解为 Concentration（即 incoherence，C ∝ 1/µ²）和 Alignment（权重-激活方向匹配度）的乘积。正交变换在数学上不可能改善 Alignment（A(Rx,WR^T)=A(x,W)），而 Alignment 在某些层上可贡献 >10 dB 的改善空间

## 引用关系

← 理论来源：[[QuIP#]]（定义 µ-incoherence，推导 RHT bound，提出"incoherence 是量化本质"）
← 实践应用：[[QuaRot]]（将 RHT incoherence processing 从 weight-only 扩展到 W+A+KV 全量化）
← 验证与扩展：[[SpinQuant]]（发现随机 RHT 的 incoherence 质量有方差，通过学习旋转消除）
← 理论扩展：[[Concentration-Alignment]]（将 incoherence 重新表述为 Concentration，并揭示被忽视的 Alignment 因素）
→ 上层概念：[[Rotation-based Quantization]]（incoherence 是旋转量化的理论基础）
→ 计算实现：[[Hadamard Transform]]（RHT 是实现 incoherence processing 的首选算法）
→ 协同技术：[[Lattice Codebooks]]（incoherence → 球形分布 → 格码本 VQ 的逻辑链）
