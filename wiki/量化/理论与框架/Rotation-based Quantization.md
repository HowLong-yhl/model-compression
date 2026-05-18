---
title: Rotation-based Quantization
aliases: [旋转量化, Rotation Quantization, Hadamard Rotation]
tags: [model-compression, quantization, concept, technique]
created: 2026-04-20
updated: 2026-04-20
sources: [A Comprehensive Evaluation on Quantization Techniques for LLMs, SpinQuant, QuIP#, QuaRot, OSTQuant, FlatQuant]
---

## 定义

Rotation-based Quantization 是一种预量化变换技术，通过将数据矩阵乘以正交矩阵（orthogonal matrix）来重新分配激活值的分布，从而大幅消除 [[Emergent Outlier Features]]，使低比特量化成为可能。它是 [[Equivalent Scaling Transformation]] 之外的另一核心预量化技术。

## 基本原理

对线性层 `Y = XW`，插入正交矩阵 O（满足 `O·O^T = I`）：

```
Y = XW = (X·O)(O^T·W) = X̂·Ŵ
```

正交变换保证了数学等价性（因为 `O·O^T = I`），但旋转后激活 X̂ 的分布发生根本性变化——原先集中在少数通道的 outlier 被"分散"到所有维度中，使得分布变得均匀，**per-channel 或 per-tensor 量化重新变得可行**。

### 为什么旋转能消除 outlier

正交矩阵的几何意义是刚性旋转（保距变换）。当 outlier 集中在少数维度时，旋转将这些"尖峰"均匀分散到所有维度——类似于在高维球面上重新分配能量。旋转不改变 tensor 的 Frobenius 范数（总"能量"不变），但改变了能量在各维度的分布。

## 方法演进

根据 [[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 的分类：

| 方法 | 旋转矩阵类型 | 是否优化 | 额外技术 |
|------|-------------|---------|---------|
| QuIP | 随机正交矩阵 | 否 | — |
| QuIP# | **随机 Hadamard 矩阵** | 否 | Lattice codebooks |
| [[QuaRot]] | 随机 Hadamard 矩阵 | 否 | 扩展到 W+A 量化 |
| [[SpinQuant]] | Hadamard 初始化 | **是（Stiefel 流形优化）** | 降低旋转引入的方差 |
| FlatQuant | 可学习仿射变换（**不严格正交**） | **是（梯度优化）** | 结合 scaling, Kronecker 分解 |
| OSTQuant | 可学习正交矩阵 | **是（梯度优化）** | 结合 scaling |
| ResQ | — | 否 | Reorder + Rotation |
| QServe | — | — | Scaling + Rotation |

### [[Hadamard Transform|Hadamard 矩阵]]的优势

大多数方法使用 Hadamard 矩阵而非通用正交矩阵，原因：
- Hadamard 矩阵元素仅为 ±1/√n，乘法可高效实现（O(n log n) 的 Fast Walsh-Hadamard Transform）
- 避免了通用矩阵乘法的 O(n²) 开销
- 在线 Hadamard 旋转仅引入约 **~8% 延迟增加**（SpinQuant 测试结果）

### 优化 vs 非优化

[[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 的实验揭示了一个重要区别：

- **有优化**：optimized rotation + scaling 效果最好，优化消除了随机旋转引入的高方差
- **无优化（大模型场景）**：单独随机旋转反而优于旋转 + calibrated scaling，因为旋转后 outlier 已被均匀化，传统 max-based calibration scaling 反而不准确

详见子概念页：
- [[Unoptimized Rotation]]：随机/固定旋转（QuIP#, QuaRot）——零数据，但有方差
- [[Optimized Rotation]]：可学习旋转（SpinQuant, OSTQuant, FlatQuant）——消除方差，稳定最优

**注意**：FlatQuant 的可学习变换不严格约束正交性（是一般可逆矩阵），但因其功能与旋转等价（重分布 outlier + 使分布变平坦），在本 wiki 的 2×2 分类中仍归入 "Rotation (Optimized)" 类别。严格正交约束仅 SpinQuant 和 OSTQuant 满足。

### Incoherence Processing（QuIP# 的理论视角）

[[QuIP#]] 提出了一个更严格的理论框架来理解旋转的作用：**[[Incoherence]]**。µ-incoherent 矩阵的最大元素满足 max|W_ij| ≤ µ·‖W‖_F / √(mn)，即没有 outlier。RHT 的 incoherence bound 为 O(√log(mn))，优于 Kronecker 方法的 O(log²)。

核心论点：各种 outlier suppression heuristic（scaling, mixed-precision）本质上都在追求 incoherence。RHT 是目前理论上最优且计算最高效的 incoherence processing 方式。

### SpinQuant 的方差分析

[[SpinQuant]] 的核心发现是 **随机旋转引入巨大性能方差**：

- 100 次随机正交旋转的 W4A4 accuracy 差异高达 **13 分**
- 随机 Hadamard 矩阵整体优于一般随机旋转（与 QuIP# 的 tighter bound 一致），但仍有 **~6 分方差**
- Cayley SGD 优化后，方差降至接近零，且稳定优于所有随机试验中的最优者

这解释了为什么同一旋转框架在不同论文中报告不同结果——选择的随机种子影响巨大。

## 与 Scaling 的关系

Rotation 和 [[Equivalent Scaling Transformation|Scaling]] 是两种互补的预量化变换：

| 维度 | Scaling | Rotation |
|------|---------|----------|
| 机制 | 调整各通道的缩放比例 | 刚性旋转改变整体分布 |
| 消除 outlier 的力度 | 中等（通道间重新平衡） | **强**（跨维度重分配） |
| 计算开销 | 零（可融入 LayerNorm/权重） | 大部分可融入权重，少量需在线计算 |
| 适用精度 | W8A8, W4A16 | **W4A4**（更低比特） |
| 代表方法 | [[SmoothQuant]], [[AWQ]] | QuaRot, SpinQuant |

最新方法（FlatQuant、OSTQuant）同时使用两者：先 rotation 消除大 outlier，再 scaling 微调通道分布。

**理论局限**：[[Concentration-Alignment]] 从 SQNR 分解角度严格证明了正交变换**在数学上不可能改善 Alignment**（A(Rx, WR^T) = A(x,W) 对任意正交 R 严格成立）。旋转仅通过中心极限定理改善 Concentration（使分布趋近 Normal），而 Alignment 是一个贡献可达 10+ dB 的独立误差因素。这解释了即使最优旋转+缩放仍存在的"最后 1-2 PPL"残差。

## FP4 下的特殊表现

[[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 发现了一个重要现象：**rotation 对 MXFP4/NVFP4 几乎无效**。原因是这些格式的极小 group size（16-32）本身就像 rotation 一样缓解了 outlier 的影响。更意外的是，rotation 后 INT4 反而优于 FP4——因为 rotation 使分布变均匀后，INT4 的均匀量化点比 FP4 的非均匀量化点更合适。

详见 [[FP4 Quantization]]。

## 旋转位置与推理开销（SpinQuant 总结）

[[SpinQuant]] 识别了 Transformer 中四个旋转插入点及其权衡：

| 旋转 | 位置 | 可融入权重？ | 适用场景 |
|------|------|-------------|---------|
| R₁ | Residual stream | ✅ | W4A8 即可获益 |
| R₂ | V → O_proj（head-wise） | ✅ | 同上 |
| R₃ | Q/K after RoPE | ❌ 需在线 Hadamard | W4A4 + 低比特 KV cache |
| R₄ | FFN down-proj 输入 | ❌ 需在线 Hadamard | W4A4 极端设定 |

- **SpinQuant_no_had**（仅 R₁+R₂）：零推理开销，适合 W4A8 / W4A16
- **SpinQuant_had**（全部 R₁~R₄）：~8% 额外延迟，适合 W4A4KV4 极端设定

## 向量量化与旋转的协同（QuIP# 路线）

[[QuIP#]] 揭示了旋转与向量量化的深层协同：

1. RHT 使权重分布变为近似**球形高斯** → 自然适合球形 codebook
2. E₈ 格是 R⁸ 中的最优球堆积（kissing number = 240）→ 用作 8 维 VQ 码本
3. Block LDLQ 将自适应舍入推广到块级 VQ → 利用格结构实现 O(1) 解码

这一"incoherence → 球形分布 → 格码本"的逻辑链，使得 2-bit weight-only 量化首次可用。

## Rotation完整数据流
![[images/Pasted image 20260421204644.png]]

## 开放问题

1. **旋转 + 激活量化 + VQ 能否统一？** SpinQuant 量化激活但用标量量化；QuIP# 用 VQ 但仅量化权重
2. **最优旋转是否模型相关？** SpinQuant 发现不同模型的最优旋转差异很大（LLaMA-3 比 LLaMA-2 更难量化）
3. **1-bit 量化的可能性**：QuIP# 证明了 scaling 规律可延伸，2-bit 已可用，理论上 1-bit 的 scaling 可能超过 2-bit
4. **KV Cache 量化中的旋转**：[[TurboQuant]]（ICML 2025）将 random rotation 专用于 KV cache 量化，并建立了 rate-distortion 理论最优性保证；RotateKV（IJCAI 2025）提出 outlier-aware adaptive rotation 进一步提升 2-bit KV cache 精度。旋转在 KV cache 场景的应用正在从全量化框架（QuaRot/SpinQuant 的 KV4）分化为独立研究方向。详见 [[KV Cache Quantization]]
5. **Low-Rank 与旋转的深层联系**：[[ShadowKV]] 发现 Pre-RoPE Key 具有低秩性——这与旋转消除 outlier 的思路高度相关。RoPE 旋转破坏了 Key 的通道稳定性和低秩结构；Pre-RoPE 处理在量化（per-channel 量化）、旋转（保持 incoherence）、低秩压缩（SVD 有效性）中都有效，形成一个统一的"Pre-RoPE 结构化特性"研究主题。详见 [[Low-Rank KV Compression]]、[[KV Cache Management]]

