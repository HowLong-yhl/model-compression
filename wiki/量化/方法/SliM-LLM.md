---
title: "SliM-LLM: Salience-Driven Mixed-Precision Quantization for Large Language Models"
aliases: [SliM-LLM, SliM-LLM+, Huang 2024]
source_type: paper
source_url: "https://arxiv.org/abs/2405.14917"
source_date: 2024-05-23
ingested: 2026-04-28
venue: ICML 2025
authors: [Wei Huang, Haotong Qin, Yangdong Liu, Yawei Li, Qinshuo Liu, Xianglong Liu, Luca Benini, Michele Magno, Shiming Zhang, Xiaojuan Qi]
affiliation: University of Hong Kong, ETH Zurich, Peking University
code: "https://github.com/Aaronhuang-778/SliM-LLM"
tags: [model-compression, quantization, method, post-training, mixed-precision, weight-only, salience, group-wise]
---

## 核心要点

SliM-LLM 提出了一种**结构化的 group-wise 混合精度**量化框架，解决了现有极低比特（≤3-bit）量化的核心矛盾——uniform precision 牺牲精度，而 element-wise mixed-precision 虽然准但对硬件极不友好（需要额外的 bitmap 或码本索引存储）。

核心洞察：LLM 中的重要权重不是随机分布的，而是呈现**空间聚簇（spatial clustering）**——沿特定通道集中出现。这使得可以在 group 粒度（而非 element 粒度）分配不同比特宽度，在保持硬件友好的 uniform group 结构的同时获得混合精度的收益。

两个关键组件：
- **Salience-Determined Bit Allocation (SBA)**：基于 group 级 salience 排序分配比特宽度，用 KL 散度（而非 MSE）优化
- **Salience-Weighted Quantizer Calibration (SQC)**：在 group 内部感知局部 salient weights，优化量化器参数以保护关键信息

SliM-LLM 以 [[GPTQ]] 为 backbone；扩展版 SliM-LLM+ 将 SBA 集成到 [[OmniQuant]] 中通过梯度优化进一步提升。

## 方法详解

### 核心洞察：Salience 的空间聚簇

**Parameter Salience** 定义（遵循 SparseGPT）：

```
δ_{i,j} = w²_{i,j} / [H⁻¹]²_{j,j}
```

其中 H 为 proxy Hessian 矩阵。作者证明（Theorem 1）：当输入中存在 outlier channel 时，Hessian 对角线 H_{p,p} 在 outlier 位置会特别大，导致对应权重的 salience 沿该通道方向聚簇。

实证观察（图 3）：在 LLaMA-7B 的第 2 层 attention projection 中，salient weights 集中在第 2100、3218、3853 通道附近形成明显的簇。这种模式在不同层中一致出现。

**关键推论**：既然 salience 按通道聚簇，那么基于 group 的 mixed-precision 就能高效捕捉这种结构，无需 element-level 的细粒度控制。

### Part 1：Salience-Determined Bit Allocation (SBA)

**目标**：给定平均比特宽度 N，为每个 quantization group 分配 (N-1)-bit、N-bit 或 (N+1)-bit。

**流程**：

1. 计算每个 group 的平均 salience：$S_g = \text{avg}(w^2_{:,g} / [H^{-1}]^2_{\text{diag}})$
2. 按 salience 排序所有 groups
3. 双指针搜索：同时增加 (N-1)-bit groups 和 (N+1)-bit groups 的数量 p，保持平均比特宽度为 N
4. 对每个 p，用 **KL 散度**（非 MSE）评估量化后 block 输出与 FP16 输出的分布差异
5. 选择使 KL 散度最小的 p*：top-p 高 salience groups 分配 (N+1)-bit，bottom-p 低 salience groups 分配 (N-1)-bit

**为什么用 KL 而不是 MSE？**

| 评价函数 | OPT-1.3B 2-bit | LLaMA-7B 2-bit |
|---------|----------------|----------------|
| MSE | 32.50 | 21.94 |
| **KL Divergence** | **30.71** | **14.58** |

KL 散度从信息熵角度保护了关键激活位置的分布，在 LLaMA 上比 MSE 降低了 33% 的 perplexity。

**搜索效率**：group size=128 时，搜索区间仅 [0, m/(128×2)]，LLaMA-7B 只需 16 次迭代。

### Part 2：Salience-Weighted Quantizer Calibration (SQC)

SBA 解决的是 group 间的全局 salience 分配，但 group 内部仍有约 1% 的离散 salient weights 会被 vanilla quantizer 忽视（因为 group 内的 non-salient weights 占多数，主导了量化参数）。

**SQC 方案**：

1. 用 salience mask 将 group 内权重分为 salient ($w_s$) 和 non-salient ($w_{us}$)
2. 分别计算两部分的量化损失 $\mathcal{L}_s$ 和 $\mathcal{L}_{us}$
3. 搜索量化器参数 τ（缩放 Δ 和 zero-point z 的微调因子），最小化加权总损失
4. **关键**：$w_s$ 和 $w_{us}$ 最终共享同一组 (Δ*, z*)，无需额外存储——SQC 仅改变了搜索目标函数，不改变存储格式

效果：vanilla quantizer 的平均绝对误差为 0.0055，SQC 降至 0.0039（降低 29%），且 salient weights（红色区域）的误差降幅尤为显著。

### 集成方式

- **SliM-LLM** = SBA + SQC + GPTQ（统计量化器 backbone）
- **SliM-LLM+** = SBA + OmniQuant LWC（梯度优化 backbone，不含 SQC 因为 LWC 已做 clipping 优化）

## 实验结果

### WikiText-2 Perplexity — 统计量化器（SliM-LLM，backbone: GPTQ）

| 比特 | 方法 | LLaMA-7B | LLaMA-13B | LLaMA-30B | LLaMA-65B | LLaMA2-7B | LLaMA3-8B |
|------|------|----------|-----------|-----------|-----------|-----------|-----------|
| 3-bit | GPTQ | 6.55 | 5.62 | 4.80 | 4.17 | 6.29 | 8.19 |
| 3-bit | AWQ | 6.46 | 5.51 | 4.63 | 3.99 | 6.24 | 8.22 |
| 3-bit | **SliM-LLM** | **6.40** | **5.48** | **4.61** | **3.99** | **6.24** | **7.16** |
| 2-bit | GPTQ | 152.31 | 20.44 | 13.01 | 9.51 | 60.45 | 210.00 |
| 2-bit | QuIP | 29.74 | 12.48 | 11.57 | 7.83 | 39.73 | 84.97 |
| 2-bit | PB-LLM | 24.61 | 17.73 | 12.65 | 7.85 | 25.37 | 44.12 |
| 2-bit | **SliM-LLM** | **14.58** | **8.87** | **7.33** | **5.90** | **16.01** | **39.66** |

2-bit 下 SliM-LLM 相比 GPTQ 降低了 **90%** 的 perplexity（LLaMA-7B: 152.31→14.58）。

### WikiText-2 Perplexity — 梯度量化器（SliM-LLM+，backbone: OmniQuant）

| 比特 | 方法 | LLaMA-7B | LLaMA-13B | LLaMA-30B | LLaMA-65B | LLaMA2-7B |
|------|------|----------|-----------|-----------|-----------|-----------|
| 3-bit | OmniQuant | 6.15 | 5.44 | 4.56 | 3.94 | 6.03 |
| 3-bit | AffineQuant | 6.14 | 5.45 | 4.59 | — | 6.08 |
| 3-bit | **SliM-LLM+** | **6.07** | **5.37** | **4.34** | **3.72** | **5.94** |
| 2-bit | OmniQuant | 9.72 | 7.93 | 7.12 | 5.95 | 11.06 |
| 2-bit | AffineQuant | 13.51 | 7.22 | 6.49 | — | 10.87 |
| 2-bit | **SliM-LLM+** | **9.68** | **7.17** | **6.41** | **5.74** | **10.87** |

SliM-LLM+ 在所有模型尺寸和比特宽度上一致优于 OmniQuant。

### Zero-shot 任务精度（2-bit）

| 模型 | 方法 | PIQA | ARC-e | ARC-c | BoolQ | HellaSwag | Winogrande | Avg. |
|------|------|------|-------|-------|-------|-----------|------------|------|
| LLaMA-7B | GPTQ | 55.49 | 31.02 | 22.17 | 53.49 | 33.84 | 41.91 | 39.65 |
| | OmniQuant | 63.63 | 43.91 | 27.32 | 58.02 | 48.78 | 52.97 | 49.11 |
| | **SliM-LLM+** | **64.96** | **45.66** | **28.67** | **64.59** | **48.86** | **53.35** | **51.02** |
| LLaMA-65B | GPTQ | 76.16 | 52.48 | 40.14 | 77.23 | 71.96 | 70.22 | 64.70 |
| | OmniQuant | 77.78 | 53.71 | 40.90 | 78.04 | 74.55 | 68.85 | 65.64 |
| | **SliM-LLM+** | **78.06** | **53.90** | **41.18** | **78.33** | **75.59** | **69.99** | **66.18** |

### 部署性能（A800 GPU）

| 配置 | Weight Memory | Running Memory | PPL↓ | Token/s |
|------|-------------|---------------|------|---------|
| FP16 LLaMA-7B | 12.6G | 14.4G | 5.68 | 69.2 |
| 3-bit GPTQ | 3.2G | 5.1G | 6.55 | 83.4 |
| 3-bit SliM-LLM | 3.2G | 5.2G | 6.40 | 79.1 |
| 2-bit GPTQ | 2.2G | 4.4G | 152.31 | 83.9 |
| 2-bit SliM-LLM | 2.2G | 4.4G | 14.58 | 79.8 |

混合精度方案在保持与 uniform 量化几乎相同的内存占用和推理速度的同时，大幅提升了精度。2-bit 下内存压缩约 **6×**。

## Calibration

- **数据集**：WikiText-2（与 GPTQ/OmniQuant 一致）
- **样本数**：128 segments × 2048 tokens
- **Group size**：默认 128
- **SBA 搜索**：每层约 16 次迭代（极低开销）
- **SQC 搜索**：τ ∈ [1-λ, 1+λ]，λ=0.1，50 步网格搜索

## 消融实验

### SBA vs ILP（Integer Linear Programming）

| 方法 | LLaMA-7B 2-bit | LLaMA-65B 2-bit |
|------|----------------|-----------------|
| ILP（整数线性规划） | 17.55 | 7.46 |
| **SBA** | **14.58** | **5.90** |

SBA 的激活感知 KL 散度优化显著优于基于权重 MSE 的 ILP 方法。

### Group Size 影响

| Group Size | 3-bit LLaMA-7B | 2-bit LLaMA-7B |
|-----------|----------------|----------------|
| 512 | 6.96 | — |
| 256 | 6.92 | — |
| 128 | 6.40 | 14.58 |
| 64 | — | 13.41 |
| 32 | — | 11.91 |

Group size 128 是精度-效率的常用平衡点，与 [[Quantization Granularity]] 的结论一致。

## 局限性

- **部署工具支持不足**：当前主流推理框架对 mixed-precision（1/2/3-bit 混合）的硬件支持仍不成熟，推理速度提升受限
- **1-bit 操作缺乏硬件支持**：SBA 分配的最低精度为 (N-1)-bit，当 N=2 时需要 1-bit 操作，目前 GPU 上效率低
- **仅做 weight-only 量化**：未扩展到 weight-activation 联合量化
- **搜索空间受限于三档精度**：仅分配 {N-1, N, N+1} 三种比特宽度，更灵活的多档分配可能带来进一步收益
- **依赖 backbone 方法**：SliM-LLM 和 SliM-LLM+ 的绝对性能受限于 GPTQ/OmniQuant 的能力上限

## 与已有知识的关联

### PTQ 方法分类中的位置

在 [[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 的两步分解框架中，SliM-LLM 的创新主要在 **量化精度分配**（一个两步分解未显式覆盖的正交维度）：

```
SliM-LLM  = GPTQ backbone + Group-wise Mixed Precision (SBA + SQC)
SliM-LLM+ = OmniQuant backbone + Group-wise Mixed Precision (SBA)
```

这是一种**元方法**——可以插入到任何 group-wise 量化方法中作为精度分配策略。

### 与其他方法的关系

- **vs [[GPTQ]]**：SliM-LLM 以 GPTQ 为 backbone，保留其 Hessian 误差补偿，在此基础上增加了 group-level 混合精度分配。2-bit 下将 GPTQ 的 PPL 从 152.31 降至 14.58
- **vs [[AWQ]]**：AWQ 的 per-channel scaling 在 2-bit 下完全失效（PPL 2.6e5），SliM-LLM 的 group-wise mixed-precision 策略则保持有效
- **vs [[OmniQuant]]**：SliM-LLM+ 将 SBA 集成到 OmniQuant 中，在所有配置上进一步提升——证明混合精度与可学习量化器正交且互补
- **vs PB-LLM / SpQR / BiLLM**：这些都是 element-wise 混合精度方法（需要额外 bitmap 存储），SliM-LLM 的 group-wise 方案硬件更友好但效果可比甚至更优
- **vs [[QuIP#]] / AQLM**：这些使用 codebook-based VQ 实现极低比特量化，与 SliM-LLM 的 uniform group 方案代表了两条不同的技术路线

### 与 Salience / Outlier 研究的联系

SliM-LLM 的 salience 聚簇发现与 [[Emergent Outlier Features]]（由 [[LLM.int8()]] 首次发现）有直接联系——outlier channel 导致了 Hessian 对角元素的极端值，进而驱动了 salience 的空间聚簇。SliM-LLM 从另一个角度利用了这个现象：不是试图消除 outlier（如旋转方法），而是给 outlier 所在的 group **分配更高精度**。

## 引用关系

← 使用的 backbone 方法：[[GPTQ]]（SliM-LLM 的量化引擎），[[OmniQuant]]（SliM-LLM+ 的梯度优化 backbone）
← 理论基础：[[Emergent Outlier Features]]（salience 聚簇的底层原因），[[Quantization Granularity]]（group-wise 量化的粒度分析）
← 相关概念：[[Weight-Only Quantization]]（SliM-LLM 的量化范式），[[Post-Training Quantization]]（两步分解框架）
→ 正交贡献：Mixed Precision 是两步分解框架未覆盖的维度，SliM-LLM 作为元方法可叠加到任何 group-wise PTQ 中
→ 应用场景：极低比特 LLM 部署（2-3 bit）、边缘设备推理、内存受限场景
