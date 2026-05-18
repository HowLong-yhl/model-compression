---
title: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
aliases: [AWQ, Lin 2023]
source_type: paper
source_url: "https://arxiv.org/abs/2306.00978"
source_date: 2023-06-01
ingested: 2026-04-20
venue: MLSys 2024 (Best Paper Award)
authors: [Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han]
affiliation: MIT Han Lab
code: "https://github.com/mit-han-lab/llm-awq"
tags: [model-compression, quantization, method, post-training, weight-only, deployment]
---

## 核心要点

AWQ 提出了一种硬件友好的 [[Weight-Only Quantization]] 方法，核心发现：LLM 中仅 **0.1%-1%** 的权重通道是"salient"的，保护这些通道能大幅降低量化误差。关键创新在于**通过激活分布而非权重本身**来识别 salient 通道，并用数学等价的缩放变换（而非硬件低效的混合精度）来保护它们。

## 方法摘要

### 核心洞察

1. **并非所有权重同等重要**：0.1%-1% 的 salient weight channels 对量化误差有不成比例的影响
2. **激活幅度是正确的重要性信号**：大激活值的通道对应的权重量化误差会被放大——应看激活分布，而非权重分布
3. **缩放优于混合精度**：保留 1% FP16 权重虽然准确但硬件低效；等价缩放在硬件上是统一格式

### 等价缩放变换

对权重组中的 salient 通道乘以缩放因子 `s > 1`，对应激活通道除以 `s`：

```
Q(w · s) · (x / s) ≈ w · x （数学等价，但量化误差更小）
```

缩放使 salient 通道的**相对**量化误差降低（`Δ'` 增长慢于 `s`），但过大的 `s` 会增大非 salient 通道的量化步长——存在最优平衡点。

### 优化搜索

- 搜索空间：`s = s_X^α`，其中 `s_X` 为 per-channel 平均激活幅度，`α ∈ [0, 1]`
- Grid search：20 个均匀步长
- 同时优化 weight clipping 以进一步降低 MSE
- **搜索开销极低**——仅需 20 步前向 MSE 评估，而 [[GPTQ]] 需要收集全部 calibration 数据的激活并计算逐层 Hessian，使得 AWQ 更快且不会过拟合 calibration 数据

## 实验结果

### WikiText-2 Perplexity (INT4, W4A16, group size 128)

| 模型 | FP16 | RTN | GPTQ | **AWQ** |
|------|------|-----|------|---------|
| LLaMA-7B | 5.68 | 5.96 | 6.22 | **5.78** |
| LLaMA-13B | 5.09 | 5.20 | 5.17 | **5.19** |
| LLaMA-65B | 3.53 | 3.67 | 3.66 | **3.62** |
| Llama-2-7B | 5.47 | 5.73 | 5.69 | **5.60** |
| Llama-2-70B | 3.32 | 3.46 | 3.42 | **3.41** |

AWQ 在所有模型尺寸上一致获得 **最低 perplexity**。

### INT3 (group size 128)

| 模型 | RTN | GPTQ | **AWQ** |
|------|-----|------|---------|
| OPT-6.7B | 23.54 | — | **11.39** |
| LLaMA-7B | 7.01 | 8.81 | **6.35** |

INT3 下差距更加悬殊，AWQ 大幅领先。

### 多模态模型 (OpenFlamingo-9B, COCO CIDEr 32-shot)

| 方法 | CIDEr | Δ vs FP16 |
|------|-------|-----------|
| FP16 | 81.70 | — |
| GPTQ INT4 | 74.98 | **-6.72** |
| **AWQ INT4** | **80.53** | **-1.17** |

AWQ 在多模态 LM 上的保真度远超 GPTQ，首次实现了多模态模型的近无损量化。

### Calibration 鲁棒性

| Calibration → Eval | AWQ PPL 退化 | GPTQ PPL 退化 |
|--------------------|-------------|---------------|
| PubMed → Enron | +0.5~0.6 | +2.3~4.9 |
| Enron → PubMed | +0.5~0.6 | +2.3~4.9 |

AWQ 不使用 backpropagation，因此**不会过拟合 calibration 分布**，对领域偏移的鲁棒性远优于 GPTQ。

### 极端低比特：INT2 (group size 64)

| 模型 | RTN | GPTQ | **AWQ + GPTQ** |
|------|-----|------|----------------|
| OPT-6.7B | 7,622 | 16.65 | **15.71** |

AWQ 与 GPTQ **正交且可组合**：AWQ 提供缩放，GPTQ 提供重建。

### TinyChat 推理系统

| 平台 | 加速比 vs FP16 HF |
|------|--------------------|
| RTX 4090 | **3.7×** |
| Jetson Orin | **3.5×** |

实现了 **Llama-2-70B 在移动/边缘 GPU** 上的部署。

## Calibration

- **数据集**：The Pile
- **样本数**：仅 **16 sequences**（GPTQ 需 128，约 10× 更少）
- **Group size**：默认 128
- 即使对多模态模型也使用纯文本 calibration 数据

## 局限性

- 缩放的 trade-off 是固有的：过度缩放（s=4）会损害非 salient 通道
- 量化函数不可微，grid search 是启发式而非最优
- 仍需 calibration（虽然仅 16 样本），不是完全 zero-shot
- 仅做 weight-only（W4A16），不降低激活计算成本
- INT2 需要与 GPTQ 组合，表明纯缩放方法有下限
- TinyChat 加速依赖特定 GPU 的自定义 CUDA kernel

## 与已有知识的关联

- 与 [[SmoothQuant]] 共享"[[Equivalent Scaling Transformation|等价缩放变换]]"核心技术——SmoothQuant 缩放激活通道，AWQ 缩放权重通道
- 在两步分解框架中属于 [[Unoptimized Scaling]] + RTN 的组合（参见 [[Post-Training Quantization]]）
- 与 [[GPTQ]] 互补：GPTQ 做 Hessian-based 重建，AWQ 做 activation-aware 缩放，**可组合使用**
- **多模态扩展**：[[VLM Quantization Best Practices]] 发现 AWQ 在 VLM 上将 80-98% 重要性集中到 LLM 组件（激活感知策略优先保护大权重矩阵），在推理型 VQA 任务上 LLM 重要性高达 98.6%。[[Q-VLM]]（NeurIPS 2024）以 AWQ 作为 baseline，发现 AWQ 在 LLaVA-7B W4A4 上 ScienceQA 仅 74.02%（因未建模跨层依赖），而 Q-VLM 达到 79.79%
- 受 [[Emergent Outlier Features]]（由 [[LLM.int8()]] 首次发现）启发：salient weights 对应高激活通道
- 出自与 SmoothQuant 相同的实验室（[[MIT Han Lab]]），共同构成了从 W8A8 到 W4A16 的完整 PTQ 方案
- 同一实验室后续推出的 SVDQuant（ICLR 2025）将 outlier 处理思想延伸到扩散模型，实现了 W4A4 量化（详见 [[Diffusion Model Quantization]]）

## 引用信息

原始来源：`raw/AWQ.md`（待录入）
