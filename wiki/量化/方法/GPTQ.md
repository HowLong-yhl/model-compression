---
title: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
aliases: [GPTQ, OPTQ, Frantar 2022]
source_type: paper
source_url: "https://arxiv.org/abs/2210.17323"
source_date: 2022-10-31
ingested: 2026-04-20
venue: ICLR 2023
authors: [Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh]
affiliation: IST Austria (ISTA), ETH Zurich
code: "https://github.com/IST-DASLab/gptq"
tags: [model-compression, quantization, method, post-training, weight-only, second-order]
---

## 核心要点

GPTQ 是第一个能在约 4 小时内将 175B 参数模型量化到 3-4 bit 且几乎无精度损失的方法。它基于近似二阶信息（Hessian 逆矩阵），在 OBQ（Optimal Brain Quantization）的基础上实现了超过 **1000× 加速**，使超大模型的 [[Post-Training Quantization]] 在实践中可行。

## 方法摘要

### 学术谱系

OBD (1990) → OBS (1992) → OBC/OBQ (2022) → **GPTQ (2022)**

### Layer-wise 量化框架

GPTQ 将全模型量化分解为逐层独立的子问题。对每一层，目标是找到量化权重 Ŵ 使得 `‖WX − ŴX‖²` 最小，其中 X 是 calibration 数据的层输入。每层的 Hessian 为 `H = 2XX^T`，所有行共享同一个 Hessian。

### 三大关键创新

**1. Arbitrary Order Insight（任意顺序洞察）**

OBQ 使用贪心顺序——每步选量化误差最小的权重。GPTQ 发现**固定顺序（如从左到右逐列）与贪心顺序效果几乎一致**，尤其在列数较多的大层中。这一洞察消除了贪心选择的开销，更重要的是使后续两个优化成为可能。

**2. Lazy Batch Updates（延迟批量更新）**

将列分成大小 B=128 的 block。block 内量化某列时只更新同 block 内的列，对后续 block 的更新累积到 block 结束后一次性传播。将大量小型 memory-bound 操作转化为少量大型 compute-bound 矩阵乘法，带来约 **100× 加速**，精度损失 <0.1% perplexity。

**3. Cholesky Reformulation（Cholesky 重构）**

预先计算 Hessian 逆矩阵的 Cholesky 分解 `H⁻¹ = LL^T`，量化过程中直接读取 Cholesky 因子的对应行，无需迭代 rank-1 更新。搭配阻尼因子 `λ = 0.01`（`H ← H + λ·diag(H)`），大幅提升数值稳定性。

### 总体复杂度

每层 `O(d_row · d_col²)`，相比 OBQ 的 `O(d_row · d_col³)` 降低一个数量级。

### 附加技术

- **Act-Order**：按 Hessian 对角线降序排列列（即按激活敏感度从高到低），使高敏感权重在有更多剩余列可补偿时优先量化
- **Group-wise Quantization**：用于极端低比特（2-3 bit），每 g=128/64/32 个连续权重共享一组量化参数

## 实验结果

### WikiText-2 Perplexity

| 模型 | FP16 | RTN 4-bit | GPTQ 4-bit | RTN 3-bit | GPTQ 3-bit |
|------|------|-----------|------------|-----------|------------|
| OPT-6.7B | 10.86 | 12.10 | **11.39** | 5,800 | 14.86 |
| OPT-13B | 10.13 | 11.32 | **10.31** | 3,400 | 11.61 |
| OPT-175B | 8.34 | 10.54 | **8.37** | 7,300 | **8.68** |
| BLOOM-176B | 8.11 | 8.37 | **8.21** | 571 | **8.64** |

OPT-175B 在 4-bit 下仅 +0.03 perplexity；3-bit 时 RTN 完全崩溃（7,300），而 GPTQ 仍保持 8.68。

### 极端量化

| 模型 | 2-bit g128 | 2-bit g32 | Ternary g8 |
|------|-----------|----------|-----------|
| OPT-175B | 9.58 | 8.94 | 9.20 |

首次证明 175B 模型在 2-bit 甚至三值量化下仍能保持可用质量。

### 推理加速

| GPU | FP16 | GPTQ 3-bit | 加速比 |
|-----|------|-----------|--------|
| A100 (5 → 1 GPU) | 230 ms | 71 ms | **3.24×** |
| A6000 (8 → 2 GPU) | 589 ms | 130 ms | **4.53×** |

**首次实现** 175B 模型在单块 A100 上进行生成式推理。

### 量化效率

| 模型 | 量化时间 | 设备 |
|------|---------|------|
| OPT-175B | ~4.2 GPU hours | 1× A100 80GB |
| BLOOM-176B | ~3.8 GPU hours | 1× A100 80GB |

## Calibration

- **数据集**：C4（Colossal Clean Crawled Corpus）
- **样本数**：128 segments × 2048 tokens
- **用途**：仅用于计算每层 Hessian `H = 2XX^T`，无需标签或 loss 计算

## 局限性

- **Weight-only**：仅量化权重，激活保持 FP16，推理需要自定义 dequantization kernel
- 在复杂推理任务（如 GSM8K 数学）上表现可能不如 perplexity 指标所示
- 小模型（<1B）精度退化较明显（OPT-125M 在 4-bit 下退化 12.5%）
- Calibration 数据的选择虽然鲁棒，但仍有一定影响
- 需要自定义 CUDA kernel 才能实现实际加速

## 与已有知识的关联

- 与 [[AWQ]] 的关系：两者都是 [[Weight-Only Quantization]] 方法，但原理不同——GPTQ 基于 Hessian 逆矩阵的二阶重建，AWQ 基于激活感知的等价缩放。两者**可以组合**使用（AWQ 提供缩放，GPTQ 提供重建），在 INT2 极端场景下效果最佳
- **混合精度扩展**：[[SliM-LLM]] 以 GPTQ 为 backbone，在 group 层面引入 salience-driven 混合精度分配（SBA + SQC），2-bit LLaMA-7B PPL 从 152.31 降至 14.58——在不改变 GPTQ 核心算法的前提下大幅提升极低比特性能
- 与 [[SmoothQuant]] 和 [[LLM.int8()]] 的区别：后两者关注 W8A8（同时量化权重和激活），GPTQ 专注于将权重压到极低比特（3-4 bit）
- **多模态扩展**：[[VLM Quantization Best Practices]] 将 GPTQ 应用于 BLIP-2 / LLaVA，发现 GPTQ 的 Hessian 方法能捕捉跨组件交互，在多模态场景下组件重要性分布更均衡（ViT 50.4% vs LLM 47.6% on COCO captioning），与 AWQ 的 LLM 集中倾向形成对比。[[Q-VLM]]（NeurIPS 2024）指出 GPTQ 的逐层独立假设在 VLM 中失效——跨层依赖使得 block-wise 联合优化必要，Q-VLM 通过熵引导分块在 W4A4 下显著优于逐层方法
- GPTQ 的 Hessian 核心思想来自经典的 OBS（Optimal Brain Surgeon, Hassibi et al. 1992）框架，经由 OBD → OBS → OBQ → GPTQ 的学术传承发展而来

## 引用信息

原始来源：`raw/GPTQ.md`（待录入）
