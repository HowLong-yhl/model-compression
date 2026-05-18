---
title: "Dissecting Quantization Error: A Concentration-Alignment Perspective"
aliases: [Concentration-Alignment, CAT, SQNR Decomposition, 量化误差分解]
source_type: paper
source_url: "https://arxiv.org/abs/2603.04359"
source_date: 2026-03-04
ingested: 2026-05-08
venue: ICML 2026
authors: [Marco Federici, Boris van Breugel, Paul Whatmough, Markus Nagel]
affiliation: Qualcomm AI Research
tags: [model-compression, quantization, concept, theory, rotation, scaling, framework]
---

## 核心要点

本文从信号处理角度提出了量化误差的 **SQNR（Signal-to-Quantization-Noise Ratio）分解理论**，将线性层的量化误差严格分解为三个可乘因素：**(i) Bitwidth**（比特宽度）、**(ii) Concentration（集中度）**——衡量权重和激活的 outlier 程度、**(iii) Alignment（对齐度）**——衡量激活主变异方向与权重有效方向的匹配程度。

关键数学结果：**正交变换（旋转）在数学上不可能改善 Alignment**（证明见方程 4）。这意味着所有基于 Hadamard/旋转的方法（QuaRot、SpinQuant 等）只能优化 Concentration，而 Alignment 是一个被完全忽视的独立误差来源，在某些层上可贡献超过 10 dB 的 SQNR 改善空间。据此提出 **CAT（block Concentration-Alignment Transforms）**，在 W4A4 下无需训练即可超越 SpinQuant（需训练），训练后匹配或超越 FlatQuant。

## 理论框架

### SQNR 分解（Theorem 2.4）

对线性层 Y = Wx，设量化后的权重为 W̃ = Q_W(W)、量化后的激活为 x̃ = Q_x(x)。SQNR 定义为：

```
SQNR(W̃x̃) = E[‖Wx‖²₂] / E[‖Wx − W̃x̃‖²₂]
```

基于标准误差去相关假设（Widrow et al., 1996），SQNR 可分解为权重量化和激活量化各自 SQNR 的**谐波和**（parallel operator）：

```
SQNR(W̃x̃) ≈ SQNR(Wx̃) ∥ SQNR(W̃x)
```

其中 a ∥ b = (1/a + 1/b)⁻¹。

进一步展开，当权重和激活使用相同比特宽度 b 时：

```
SQNR(W̃x̃) ≈ 12 · N(b)² · (C(x) ∥ C(W)) · A(x, W)
               ─────────   ───────────────   ────────
               Bitwidth     Concentration    Alignment
```

其中 N(b) 为 b-bit 量化的区间数。每增加 1 bit，SQNR 增加约 6 dB（因子 4）。

### 因素一：Concentration（集中度）

**精确定义**：

激活 Concentration：

```
C(x) = E[‖x‖²₂] / (d · E[max_i x_i²])
```

即平均平方范数与 d 倍最大分量平方的比值。权重 Concentration 类似，为各行 ℓ₂ 范数平方之和与各行 range 平方之和的比值。

**性质**：

- **尺度不变**（scale-invariant）：仅取决于分布形状，不取决于幅度
- 与 **kurtosis（峰度）** 强相关：kurtosis 考虑 norm-4 与 norm-2 的比值，concentration 考虑 norm-2² 与 norm-∞² 的比值
- 重尾分布（存在 outlier）→ 低 concentration；所有值相同 → concentration 趋于无穷
- 对称量化下最小值为 1/4（−6 dB），非对称量化下最小值为 1/2（−3 dB）

**参考分布的 Concentration**：

- **Normal 分布**：d 维正态的 concentration ≈ d/(2 ln d)，随维度增大而增大
- **Laplace 分布**：比 Normal 低，因为更重尾
- 论文发现：LLM 激活的 concentration 通常**比 Laplace 还差**（更重尾），而权重介于 Laplace 和 Normal 之间

**与 [[Incoherence]] 的关系**：Concentration 本质上是 [[Incoherence]] 的 SQNR 语言表述。µ-incoherent 矩阵满足 max|W_ij| ≤ µ · ‖W‖_F/√(mn)，等价于 C(W) ≥ 1/µ²。两者衡量同一现象——能量在各元素间的均匀程度。

**现有方法对 Concentration 的影响**：

| 方法 | 机制 | 对 Concentration 的影响 |
|------|------|------------------------|
| Hadamard 旋转 | 中心极限定理效应，混合通道使分布趋近 Normal | **大幅提升**（尤其对激活，可改善 >10 dB） |
| Channel Scaling（[[SmoothQuant]]） | 将激活 outlier 迁移到权重 | 改善激活 concentration，但**恶化**权重 concentration |
| 非对称量化 | 偏移使量化范围匹配实际分布 | 对非对称分布有效（如 ReLU 后），对零中心分布无效 |

关键观察（Figure 4）：Channel scaling 会显著恶化权重 concentration，这是 SmoothQuant 在低 bit 下表现受限的重要原因。而 Hadamard 旋转同时改善权重和激活的 concentration，使两者都接近 Normal 分布水平。

### 因素二：Alignment（对齐度）

**精确定义**：

```
A(x, W) = E[‖Wx‖²₂] / (‖W‖²_F · E[‖x‖²₂])
```

直觉：A(x,W) 衡量给定激活分布下，权重矩阵**实际产生的输出能量**相对于**最大可能输出能量**（如果激活在所有方向均匀分布）的比值。

**性质**：

- **尺度不变**
- **对旋转不变**（关键性质！）：对任意正交矩阵 R，A(Rx, WR^T) = A(x, W)——这是严格的数学等式（方程 4），意味着**旋转在原理上不可能改善 Alignment**
- 高 alignment = 激活的主变异方向恰好是 W 能产生大输出的方向 → 信号强 → SQNR 高
- 低 alignment = 激活主要在 W 的"低效"方向变化 → 信号弱 → SQNR 低

**Alignment 的最大值**：

```
max_M A(Mx, WM⁻¹) = (Σᵢ λᵢ²) / (Σᵢ λᵢ)²
```

其中 λᵢ 为 Σ_y = W·Σ_x·W^T 的特征值。当所有特征值相等时（输出在各方向等能量），alignment 达到最大值 1/d。

**论文的关键观察（Figure 5）**：许多层（特别是 down_proj、o_proj、v_proj）的 alignment 远低于可达最大值——差距可达 **>10 dB**。这 10 dB 相当于同时增加权重和激活各 ~2 bit 的效果（log₂√k ≈ 1.7 bit when k = 10 dB ≈ 10×）。

### SQNR 因素间的交互

论文发现实际 LLM 中 r(x,W) = SQNR(Wx̃)/SQNR(W̃x) < 1（激活量化比权重量化更难）。这意味着：

- 改善激活 concentration 比改善权重 concentration 收益更大
- 增加激活 bitwidth 比增加权重 bitwidth 效果更好（Figure 3 的非对称 iso-line）
- Alignment 对两者（权重 SQNR 和激活 SQNR）同时起乘法作用，是一个"全局放大器"

## 方法：CAT（Concentration-Alignment Transforms）

### 两步设计

CAT 采用**顺序两步构造**——先优化 alignment，再优化 concentration：

**Step 1：最大化 Alignment（解析解）**

寻找变换 M̂ 使得 A(M̂x, WM̂⁻¹) 最大化。论文证明最优解为**矩阵几何均值**：

```
M̂ = (Σ_w # Σ_x⁻¹)^(1/2)

其中 Σ_x = E[xx^T]，Σ_w = W^T W
A # B = A^(1/2) (A^(-1/2) B A^(-1/2))^(1/2) A^(1/2)  （矩阵几何均值）
```

直觉：M̂ 将激活坐标系旋转+缩放到一个"最优对齐"位置，使得变换后的激活的主变异方向恰好与变换后权重的高效方向匹配。

**Step 2：最大化 Concentration（Hadamard 旋转）**

在 M̂ 之上复合 Hadamard 矩阵 H 以最大化联合 concentration：

```
T̂ = H · M̂
```

Hadamard 旋转不影响 Step 1 获得的最优 alignment（因为 alignment 对旋转不变！），但大幅提升 concentration。因此两步是**解耦的**——先做 alignment 不影响后续 concentration 优化。

### Block-Diagonal 近似

全矩阵 M̂ 需要昂贵的在线矩阵乘法。实际中使用 block-diagonal 近似：

```
M̂_block^k = Diag([M̂₁, ..., M̂_{d/k}])

每个 M̂ᵢ ∈ R^{k×k}，在 k 维子空间上独立优化 alignment
最终变换：T̂_block^k = H · M̂_block^k
```

论文实验中 k = 128（与 FlatQuant 相同的计算开销级别）。

**特殊情况 k=1**：退化为对角矩阵 M̂_block^1 = Diag(m)，其中：

```
m_i = √(E[x_i²] / Σ_j w_ij²)
```

这本质上就是 channel scaling（类似 SmoothQuant），但使用了 alignment-optimal 的公式（基于平方均值），而非 SmoothQuant 的 max-based 公式。论文发现 SmoothQuant 的 max-ratio 公式**不一定在所有层上改善 alignment**（Figure 5），而 CAT k=1 则保证最优。

### 训练模式（可选）

CAT 支持两种模式：

- **无训练（CAT untrained）**：直接使用 calibration 数据估计 Σ_x 和 Σ_w，解析计算 T̂
- **有训练（CAT trained）**：在解析解初始化基础上，用 block-level 梯度训练进一步优化（类似 SpinQuant/FlatQuant 的训练流程）

关键发现：**CAT 无训练版本已超越 SpinQuant 有训练版本**（Table 1），说明正确的变换结构（同时优化 alignment）比优化不完整的目标（仅优化 concentration）更重要。

## 实验结果

### W4A4 量化 WikiText-2 Perplexity（RTN 设置）

| 方法 | 训练 | Llama2-7B | Llama3-8B | Llama3.2-1B-it | Ministral-8B-it | Qwen3-8B |
|------|------|-----------|-----------|---------------|----------------|----------|
| FP16 | — | 5.47 | 6.14 | 13.16 | 6.97 | 9.73 |
| 无变换 | ✗ | 1621.86 | 321.22 | 412.51 | 63.59 | 23254.67 |
| SmoothQuant | ✗ | 322.44 | 233.61 | 304.22 | 51.35 | 5047.04 |
| QuaRot | ✗ | 9.31 | 10.99 | 29.89 | 9.12 | 13.69 |
| **CAT (block)** | **✗** | **6.11** | **8.29** | **18.25** | **8.07** | **10.82** |
| SpinQuant | ✓ | 6.96 | 8.87 | 20.16 | 8.72 | 10.14 |
| FlatQuant | ✓ | 6.05 | 7.61 | 17.63 | 7.95 | 10.69 |
| **CAT (block)** | **✓** | **5.95** | **7.33** | **16.66** | **7.88** | **10.50** |

关键观察：

1. **SmoothQuant 在 W4A4 下几乎完全失效**（PPL 仍在数百级别），因为它恶化了权重 concentration
2. **CAT 无训练 > SpinQuant 有训练**：正确的变换结构比优化一个不完整的目标更有价值
3. **CAT 有训练 ≈ FlatQuant**：两者在大部分模型上接近，CAT 在 Llama3-8B 上明显更优

### W4A4 量化 WikiText-2 Perplexity（GPTQ 设置）

| 方法 | 训练 | Llama2-7B | Llama3-8B | Llama3.2-1B-it |
|------|------|-----------|-----------|---------------|
| QuaRot | ✗ | 6.22 | 8.21 | 19.97 |
| **CAT (block)** | **✗** | **6.01** | **7.60** | **17.89** |
| SpinQuant | ✓ | 6.26 | 7.88 | 18.16 |
| FlatQuant | ✓ | 5.94 | 7.25 | 16.34 |
| **CAT (block)** | **✓** | **5.97** | **7.25** | **16.33** |

GPTQ 设置下，CAT 有训练与 FlatQuant 完全持平。GPTQ 对 QuaRot/SmoothQuant 帮助大，但对 CAT/FlatQuant 帮助小——因为后两者的 learnable clipping 已部分覆盖了 GPTQ 的贡献。

### SQNR 改善可视化（Figure 6）

在 Qwen v3 8B 上，CAT 变换后的 W4A4 SQNR 在**几乎所有层上超越了未变换的 W6A6**——等效于"免费"获得了 2 bit 的精度提升。改善最大的是 MLP 层（gate_proj、up_proj、down_proj），这些层未变换时 concentration 和 alignment 都很差。

## Calibration

- **数据集**：DCLM-edu（非 WikiText，避免评估偏差）
- **样本数**：128 sequences × 2048 tokens
- **用途**：估计每层的激活协方差 Σ_x = E[xx^T]，计算 alignment-optimal 变换 M̂

## 局限性

- SQNR 分解基于**均匀整数量化 + 可忽略 clipping error** 假设；对非均匀量化（codebook/VQ）和严重 clipping 的场景适用性有限
- 全矩阵 M̂ 是理论最优但实践中不可行（需在线矩阵乘法）；block-diagonal 是近似，block 大小 k 是 alignment 质量与计算效率的权衡
- 在某些有极端 outlier 的层（如 BOS token 主导的层），SQNR 近似与实际值有偏差（Figure 2 中的 layer.1.mlp.down_proj）
- 对 FP8/FP4 等浮点量化格式的适用性未探讨（论文聚焦均匀整数量化）
- 理论框架仅适用于线性层；attention 中的 softmax 非线性影响未建模

## 与已有知识的关联

- **与 [[Incoherence]] 的关系**：Concentration 是 incoherence 的 SQNR 语言重新表述（C ∝ 1/µ²）。本文揭示了 incoherence 只解释了量化误差的一半——另一半是 Alignment
- **与 [[Rotation-based Quantization]] 的关系**：本文给出了旋转"为何有效但不够"的严格证明——旋转改善 concentration（通过中心极限定理使分布趋近 Normal），但**在数学上不可能改善 alignment**（方程 4）。这解释了 QuaRot/SpinQuant 的精度天花板
- **与 [[Equivalent Scaling Transformation]] / [[SmoothQuant]] 的关系**：Channel scaling 是 CAT 在 block size k=1 时的特例。但 SmoothQuant 使用 max-ratio 公式（非最优），且会恶化权重 concentration——这解释了其在 W4A4 下的失败（PPL 300+）。CAT k=1 使用 alignment-optimal 的平方均值公式
- **与 [[FlatQuant]] 的关系**：FlatQuant 的 Kronecker 分解可学习仿射变换隐式优化了 concentration + 部分 alignment。CAT 提供了显式框架和解析解，无需训练即可接近 FlatQuant 性能
- **与 [[Quantization Insights]] 的关系**：
  - Insight 1（Rotation + Scaling 最优组合）：在 CAT 视角下，这是因为两者联合近似覆盖了 Concentration，但仍未优化 Alignment
  - Insight 4（权重对称 + 激活非对称）：非对称量化改善 concentration（减小 quantization scale），对零中心分布无效——这正是 Concentration 定义所预测的
- **与 [[Post-Training Quantization]] 两步分解框架的关系**：两步分解按"变换类型"分类（Rotation/Scaling），本文按"优化目标"分类（Concentration/Alignment）——正交互补的两个视角。CAT 的两步（alignment → concentration）与两步分解（Step 1 变换 → Step 2 补偿）是不同层面的"两步"

## 引用信息

```bibtex
@inproceedings{federici2026dissecting,
  title={Dissecting Quantization Error: A Concentration-Alignment Perspective},
  author={Federici, Marco and van Breugel, Boris and Whatmough, Paul and Nagel, Markus},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
