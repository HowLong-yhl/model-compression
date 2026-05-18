---
title: LLM 量化实验 Insights
aliases: [Quantization Insights, 量化实验发现, Experimental Findings]
tags: [model-compression, quantization, concept, insights, experimental]
created: 2026-04-20
updated: 2026-04-20
sources: [A Comprehensive Evaluation on Quantization Techniques for LLMs, SpinQuant, QuIP#, QuaRot, OSTQuant, FlatQuant]
---

## 概述

本页面汇总了来自多篇量化论文的关键实验发现与 insights，特别是 [[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 在统一实验条件下的系统评测。这些发现不属于某一具体方法，而是揭示了**量化本身的规律和原则**，对理解和设计新方法有指导意义。

---

## Insight 1：优化后的旋转+缩放是最优预量化组合

**来源**：综述论文 Table VIII, [[OSTQuant]] QSUR 理论

### 实验证据（LLaMA-3.2-1B, W4A4, per-channel）

| 预量化变换组合 | WikiText2 PPL |
|--------------|---------------|
| 无预处理 + GPTQ | >20 |
| 随机旋转 + GPTQ | 12.95 |
| 优化旋转 + GPTQ | 略优于 12.95 |
| **优化旋转 + 优化缩放 + GPTQ** | **11.80** |
| 优化旋转 + 优化缩放 + GPTQ + Low-rank | **11.73** |

### 为什么组合最优

[[OSTQuant]] 从理论上证明了这一点——QSUR（Quantization Space Utilization Rate）只有在 **rotation（消除 outlier 方向）+ scaling（消除特征值不均）** 同时作用时才能达到最大值。单独 rotation 改变了数据的"形状"但不调整"大小"；单独 scaling 调整"大小"但不改变"形状"。

---

## Insight 2：未优化时，旋转后不要做 calibrated scaling

**来源**：综述论文 Table VI

### 实验证据（LLaMA-3.2-1B, W4A4, per-channel, 无优化）

| 组合 | WikiText2 PPL |
|------|---------------|
| 仅随机旋转 | 20.31 |
| 随机旋转 + calibrated scaling | **20.96（更差！）** |

### 机制解释

随机旋转使 outlier 均匀分散到所有通道后，分布变为近似均匀高斯。此时传统 calibration scaling（基于 `max(|X_j|)^α / max(|W_j|)^{1-α}`）的假设不再成立——它假设某些通道有明显更大的幅度需要被平衡，但旋转后**所有通道幅度接近相等**，公式反而引入了不必要的非均匀性。

### 指导意义

这说明 [[Unoptimized Scaling]] 与 [[Unoptimized Rotation]] 的组合需要谨慎——**要么都不优化（但只用旋转），要么都优化**。中间状态（优化一个不优化另一个）反而可能适得其反。

---

## Insight 3：GPTQ 优于 Low-rank 补偿，但二者可叠加

**来源**：综述论文 Table VIII

### 实验证据（LLaMA-3.2-1B, W4A4, 最优预量化变换）

| 误差补偿方式 | per-channel PPL | per-group PPL |
|-------------|----------------|---------------|
| 仅 Low-rank | 14.58 | — |
| 仅 GPTQ | 11.80 | 10.89 |
| **GPTQ + Low-rank** | **11.73** | **10.87** |

### 机制解释

- **GPTQ** 直接在 Hessian 信息指导下最小化每层的量化代理损失，与 task loss 高度相关
- **Low-rank 补偿** 基于权重量化误差的 SVD 分解，捕捉的是权重空间的主要误差方向，但不直接对应 loss landscape
- 两者**互补**：GPTQ 修正主要误差后，Low-rank 进一步修正 GPTQ 未能覆盖的残差方向

### Low-rank 的推理开销

r=32 时：额外存储约 2.3%（FP16）/ 9.2%（量化后），额外计算约 3.1% FLOPs。适合对精度敏感且可接受微小开销的场景。

---

## Insight 4：激活非对称量化极其重要，权重非对称几乎无用

**来源**：综述论文 Table XI

### 实验证据（LLaMA-3.2-1B, W4A4, 旋转+缩放+GPTQ）

| 量化对称性设置 | WikiText2 PPL | ARC-C (%) |
|--------------|---------------|-----------|
| 完全对称 | 12.73 | 31.0 |
| 仅权重非对称 | 12.70 | 31.7 |
| **仅激活非对称** | **11.80** | **33.3** |
| 全部非对称 | 11.70 | 33.7 |

### 机制解释

- **激活分布天然非对称**：LLM 的激活值（尤其经过 ReLU/SiLU 后）有明显的非零均值，对称量化会浪费一半的量化范围
- **权重分布天然对称**：大多数层的权重均值接近零，对称量化已经能很好覆盖分布

### 实用建议

**权重对称 + 激活非对称** 是最佳工程折中：
- 获得了 ~93% 的全非对称收益（PPL 11.80 vs 11.70）
- 避免了权重非对称化带来的反量化计算开销（需要存储和减去 zero-point）

---

## Insight 5：量化粒度是精度-存储的连续权衡

**来源**：综述论文 Table XII, [[Quantization Granularity]]

### 核心关系

| Group size | 额外存储（bits/weight） | 效果趋势 |
|-----------|----------------------|---------|
| per-channel (=seq_len) | ~0 | 基线 |
| group-512 | 0.0625 | 微小提升 |
| group-256 | 0.125 | 小提升 |
| group-128 | 0.25 | **常用平衡点** |
| group-64 | 0.5 | 显著提升 |
| group-32 | 1.0 | 接近上限 |

### 指导意义

- Group-128 是工业界最常用的配置（GPTQ/AWQ 默认值）
- 当存储预算允许时，group-64 带来显著收益
- Group-32 以下的收益递减（因为每个 group 内样本太少，统计不稳定）

---

## Insight 6：FP4 在粗粒度下远优于 INT4，但细粒度下差距消失

**来源**：综述论文 Table XIII

### 实验证据（无预量化变换）

| 格式 | per-channel PPL | group-16 PPL |
|------|----------------|--------------|
| INT4 | 256.97 (RTN) | 接近 FP4 |
| FP4 | 135.68 | 接近 INT4 |

### 机制解释

- **粗粒度下**：FP4 的非均匀量化点天然适配权重/激活的正态分布（中心密集、尾部稀疏），而 INT4 的均匀量化点在长尾分布下浪费 bin
- **细粒度下**：每个 group 的局部分布范围很小，INT4 的均匀 bin 已经足够覆盖。FP4 的非均匀优势在局部均匀分布面前消失

### 关键推论

FP4 的优势**来源于它对全局非均匀分布的适配能力**。当分组足够细时（group ≤ 32），局部分布趋于均匀，FP4 的核心优势被 group 本身"替代"了。

---

## Insight 7：旋转对 FP4 格式几乎无效——暗示需要全新预处理范式

**来源**：综述论文 Table XIII 扩展分析

### 实验现象

- 对 INT4：旋转将 per-channel PPL 从 256.97 大幅降至 ~13（改善 20×）
- 对 MXFP4/NVFP4：旋转几乎不改善 PPL（甚至微略恶化）

### 机制解释

MXFP4/NVFP4 使用极小 group size（16-32），**小 group 本身就像旋转一样缓解了 outlier 的影响**——每个 group 有独立的 scaling factor，outlier 只影响 16-32 个值而非整个 channel。旋转做的"全局分散 outlier"在已经做了"局部隔离 outlier"的 FP4 上没有边际收益。

### 更意外的发现

**旋转后 INT4 反而优于 FP4**：旋转使分布变为均匀高斯后，INT4 的均匀量化点比 FP4 的非均匀量化点更匹配（均匀分布用均匀 bin 最优）。这是一个完美的理论验证——量化格式的最优性取决于数据分布的形状。

### 开放问题

FP4 时代需要什么样的预处理？这是当前最重要的开放研究方向之一：
- 旋转无效 → 传统 outlier suppression 思路不再适用
- Scaling 仍有效（FP4 的 group-level scaling 与 channel-level scaling 正交）
- 可能方向：优化 group 内的数据排列？自适应 group 划分？

---

## Insight 8：两步分解框架的贡献是累加且互补的

**来源**：综述论文 Section V 全局结论

### 最优配置公式

```
Best W4A4 = Optimized Rotation + Optimized Scaling + GPTQ + Low-rank(r=32)
```

### 多模型验证（综述论文 Table IX）

| 模型 | BF16 PPL | 最优 W4A4 PPL | 差距 |
|------|----------|--------------|------|
| LLaMA-3.2-1B | 9.76 | 11.73 | +2.0 |
| LLaMA-3.2-3B | 7.82 | 8.90 | +1.1 |
| Qwen2.5-3B | 8.03 | 9.22 | +1.2 |
| LLaMA-3.1-8B | 6.24 | 7.23 | +1.0 |
| LLaMA-3.1-70B | 2.81 | 4.41 | +1.6 |

### Scaling Law

- 模型越大，绝对 PPL 差距通常越小（1B: +2.0 → 3B: +1.1 → 8B: +1.0），但 70B（+1.6）反而大于 8B，说明超大模型可能面临新挑战（更多层的误差累积？更多 outlier pattern？）
- MoE 模型（Qwen1.5-MoE-A2.7B）需要更轻量策略（全量优化计算代价过高）

---

## Insight 9：推理开销的实用参考

**来源**：综述论文 Section V-E, [[SpinQuant]]

| 组件 | 推理开销 | 备注 |
|------|---------|------|
| 缩放（Scaling） | **零** | 融入权重或 LayerNorm |
| 旋转（大部分） | **零** | 融入权重 |
| 旋转（在线 Hadamard） | **~8% 延迟** | 仅 W4A4 极端设定需要 |
| GPTQ | **零** | 直接修改量化权重 |
| Low-rank (r=32) | ~2.3% 额外内存, ~3.1% FLOPs | 可选的精度提升手段 |
| FlatQuant Kronecker 变换 | ~2.6% FLOPs | Triton kernel 融合后 |

### 指导意义

W4A4 量化的总推理开销极低——绝大部分预处理都可融入权重。唯一的运行时开销来自少量在线 Hadamard 变换和可选的 low-rank 分支。这使得 W4A4 量化在部署中具有很高的性价比。

---

## 跨 Insight 的统一视角

上述 insights 可归纳为三个元原则：

1. **分布匹配原则**：量化格式（INT/FP）、量化粒度（group size）、预量化变换（rotation/scaling）本质上都在追求**数据分布与量化网格的最佳匹配**。旋转使分布变均匀 → INT4 更优；不旋转时分布是正态 → FP4 更优。

2. **互补组合原则**：不同技术解决不同维度的问题（outlier 方向 vs 幅度不均 vs 量化误差补偿），它们的贡献是累加的。最优方案是多技术组合，而非单一技术的极致。

3. **优化一致性原则**：组合中的各组件要么都不优化（保持统计性质的一致性），要么都优化（联合端到端学习）。混合使用（优化旋转+未优化缩放，或反之）可能打破组件间的兼容性。
