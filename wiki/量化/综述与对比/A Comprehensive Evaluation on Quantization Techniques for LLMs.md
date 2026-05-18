---
title: "A Comprehensive Evaluation on Quantization Techniques for Large Language Models"
aliases: [Liu et al. 2025 Survey, 量化综合评测]
source_type: paper
source_url: "https://arxiv.org/abs/2507.17417"
source_date: 2025-07-23
ingested: 2026-04-20
venue: arXiv preprint
authors: [Yutong Liu, Cairong Zhao, Guosheng Hu]
affiliation: Tongji University / University of Bristol
tags: [model-compression, quantization, survey, benchmark, W4A4, FP4, post-training]
---

## 核心要点

这是一篇系统性的 LLM 量化评测论文，其核心贡献在于：

1. 提出了一个**两步分解框架**（Two-Step Decomposition），将现有 PTQ 方法统一拆解为"预量化变换"和"量化误差补偿"两步，清晰揭示了方法间的内在联系
2. 在**同一实验条件**（W4A4 精度）下对各方法组合进行公平对比——这在此前的研究中严重缺乏
3. 前瞻性地研究了 NVIDIA Blackwell 架构的 **FP4 格式**（[[FP4 Quantization|MXFP4]] 和 [[FP4 Quantization|NVFP4]]），是最早系统评测 FP4 量化的工作之一

## 两步分解框架

### Step 1: 预量化变换（Pre-quantization Transformation）

在量化之前对权重/激活进行变换，使数据分布更适合量化。包括三种核心技术：

**Shifting（平移）**：将激活的各通道中心对齐，消除不对称性。代表方法：Outlier Suppression+、OmniQuant。

**Scaling（缩放）**：对激活通道做缩放以消除 outlier，对应权重做逆缩放保持等价。代表方法：[[SmoothQuant]]、[[AWQ]]、OmniQuant、FlatQuant、OSTQuant。参见 [[Equivalent Scaling Transformation]]。

**Rotation（旋转）**：将数据矩阵乘以正交矩阵，大幅改变激活分布并消除 outlier。代表方法：QuIP、QuIP#、[[QuaRot]]、[[SpinQuant]]、FlatQuant、OSTQuant。参见 [[Rotation-based Quantization]]。

### Step 2: 量化误差补偿（Quantization Error Mitigation）

量化后补偿引入的误差。包括两种策略：

**Self-compensation（自补偿）**：直接在原始权重上补偿。代表方法：[[GPTQ]]（基于 Hessian 的逐层 MSE 最小化）、OBQ、QuIP、GPTAQ、GQuant。

**Low-rank compensation（低秩补偿）**：引入低秩辅助分支 `Y = X·dq(Wq) + X·A·B` 来近似量化误差。代表方法：ZeroQuant-v2、LQER、CALDERA、QERA。

### 现有方法的两步分解

| 方法 | 预量化变换 | 误差补偿 |
|------|-----------|---------|
| [[SmoothQuant]] | Scaling | RTN |
| [[AWQ]] | Scaling | RTN |
| [[GPTQ]] | — | GPTQ |
| Outlier Suppression+ | Shifting + Scaling | RTN |
| QuIP | Rotation | GPTQ |
| [[QuaRot]] | Rotation | GPTQ |
| [[SpinQuant]] | Rotation | GPTQ |
| [[OmniQuant]] | Shifting + Scaling | GPTQ |
| [[Atom]] | Reorder | GPTQ |
| FlatQuant | Scaling + Rotation | RTN（原论文）/ GPTQ（可选） |
| OSTQuant | Scaling + Rotation | GPTQ |
| LQER | Scaling | Low-rank |
| ZeroQuant-v2 | — | Low-rank |

## 主要实验发现

### 1. 预量化变换

**有优化时**：optimized rotation + optimized scaling 效果最好。优化后的 scaling 可以微调旋转后 tensor 的 inter-channel 幅度分布，进一步提升效果。

**无优化时（大模型场景）**：单独使用 random rotation 优于 rotation + calibrated scaling。原因是旋转后 outlier 已被均匀化，基于 max 值的 calibration scaling 反而不准确。

### 2. 量化误差补偿

- GPTQ 通常优于单独使用 low-rank（因为 low-rank 方法基于权重误差的 SVD，未直接分析 loss error）
- **GPTQ + Low-rank 组合可以进一步提升**：per-channel 下 PPL 从 11.8 降到 11.73，per-group 下从 10.89 降到 10.87
- 两步的贡献是**可加的、互补的**

### 3. 量化配置

**对称 vs 非对称**：对激活使用非对称量化收益显著，对权重收益很小。因为激活分布天然不对称，而权重分布接近关于零对称。推荐配置：权重对称 + 激活非对称。

**粒度**：粒度越细性能越好，但存储开销增大。Group-128 是常用平衡点。详见 [[Quantization Granularity]]。

### 4. FP4 格式（核心新发现）

**FP4 vs INT4**（无预量化变换时）：per-channel 设置下 FP4 显著优于 INT4（因为 FP4 的非均匀分布更适配权重/激活的正态分布）。但在细粒度设置下差异很小。

**MXFP4 vs NVFP4**：NVFP4 更优，因为它使用更小的 group size（16 vs 32）和更精确的 scaling factor 格式（E4M3 vs E8M0）。

**关键发现：Rotation 对 FP4 几乎无效**。MXFP4/NVFP4 的小 group size 本身就像 rotation 一样缓解了 outlier，因此 rotation 的边际收益极小甚至为负。**Rotation 后 INT4 反而优于 FP4**——因为 rotation 消除 outlier 后分布变得均匀，INT4 在处理均匀分布时更强。

**Scaling factor 格式影响巨大**：E4M3 + per-tensor FP32 (NVFP4) 远优于 E8M0 (MXFP4)。将 scaling factor 降到 4-bit 会导致严重退化。

### 5. 跨模型验证

| 模型 | BF16 PPL | W4A4 PPL | 0-shot Avg |
|------|----------|----------|-----------|
| LLaMA-3.2-1B | 9.76 | 11.73 | 49.06% (vs 54.78%) |
| LLaMA-3.2-3B | 7.82 | 8.90 | 58.93% (vs 63.53%) |
| LLaMA-3.1-8B | 6.24 | 7.23 | 63.82% (vs 69.40%) |
| LLaMA-3.1-70B | 2.81 | 4.41 | 72.50% (vs 76.35%) |
| Qwen2.5-3B | 8.03 | 9.22 | 58.64% (vs 67.27%) |

使用最优配置（optimized rotation + scaling + GPTQ + low-rank），大模型的 W4A4 量化已相当实用。

## 推理成本分析

| 组件 | 推理开销 |
|------|---------|
| Scaling | 融入权重，**零开销** |
| Rotation（大部分） | 融入权重，**零开销** |
| Rotation（少量在线） | Online Hadamard，~8% 延迟增加 |
| GPTQ | 直接修改权重，**零开销** |
| Low-rank (r=32) | ~2.3% 额外内存（FP16），~3.1% 额外计算 |

## 与已有知识的关联

- 为 wiki 中的 [[GPTQ]]、[[SmoothQuant]]、[[AWQ]] 提供了统一的分类框架——它们都可以被拆解为两步
- 补充了 [[Post-Training Quantization]] 页面中缺少的 W4A4 精度配置的实验数据
- 引入了 [[Rotation-based Quantization]] 这一重要新分支（QuaRot、SpinQuant 等），是 [[Equivalent Scaling Transformation]] 之外的另一核心预量化技术
- 开辟了 [[FP4 Quantization]] 这一全新领域，与 NVIDIA Blackwell 架构直接相关
- 验证了 [[Emergent Outlier Features]] 的核心地位——所有预量化变换本质上都在对抗 outlier
- 方法间的可组合性与我们已记录的 AWQ + GPTQ 组合一致，并扩展到更多组合
- 与 [[Low-bit Model Quantization Survey]] 互补：本文聚焦 LLM PTQ 的两步分解统一框架，后者以更广视角覆盖 CNN/ViT/扩散模型量化的 8 大分类

## 引用信息

原始来源：`raw/A Comprehensive Evaluation on Quantization Techniques for LLMs.pdf`
