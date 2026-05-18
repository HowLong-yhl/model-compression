---
title: "Low-bit Model Quantization for Deep Neural Networks: A Survey"
aliases: [Low-bit Quantization Survey, Liu et al. Survey]
source_type: survey
source_url: "https://github.com/Kai-Liu001/Awesome-Model-Quantization"
source_date: 2025-01-01
ingested: 2026-04-23
venue: arXiv / IEEE (under review)
authors: [Kai Liu, Qian Zheng, Kaiwen Tao, Zhiteng Li, Haotong Qin, Wenbo Li, Yong Guo, Xianglong Liu, Linghe Kong, Guihai Chen, Yulun Zhang, Xiaokang Yang]
affiliation: Shanghai Jiao Tong University / Beihang University / HKU
code: "https://github.com/Kai-Liu001/Awesome-Model-Quantization"
tags: [model-compression, quantization, survey, taxonomy, diffusion-model, PTQ, QAT, mixed-precision]
---

## 核心要点

这篇综述系统梳理了近五年低比特量化在深度神经网络上的进展，覆盖 **200+ 篇论文**，将所有量化方法分为 **8 大类别、24 个子类别**。与本 wiki 已录入的 [[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 相比，本综述的覆盖面更广——不限于 LLM，还涵盖 CNN、ViT、扩散模型等；分类角度也不同——按**核心技术手段**而非两步分解框架分类。

## 分类体系

```
8 Main Categories
├── 3.1 Better s and z (量化范围优化)
│   ├── 3.1.1 Quantization Loss (PTQ4ViT, RAPQ, MREM, LQER, LSQ+)
│   └── 3.1.2 Task Loss (PD-Quant, LeanQuant, CBQ, Adalog)
├── 3.2 Metric and Mechanism (度量与机制)
│   ├── 3.2.1 Oscillation Reduction (OFQ, MRECG, BLAQ)
│   ├── 3.2.2 Loss Function (PD-Quant, ZeroQuant, NWQ, BSQ)
│   └── 3.2.3 Other Enlightening Mechanisms (PEQA, GPTQ, RPTQ)
├── 3.3 Mixed Precision (混合精度)
│   ├── 3.3.1 Rough Allocation (OWQ, HAWQ, SqueezeLLM)
│   ├── 3.3.2 Adaptive Allocation (NIPQ, FracBits, BSQ, SDQ, SPARQ)
│   ├── 3.3.3 Search Algorithm (HAWQ-V3, BRECQ, ZeroQ)
│   └── 3.3.4 Others
├── 3.4 Redistribution (分布重分配)
│   ├── 3.4.1 Bias
│   ├── 3.4.2 Distribution Uniformization (KURE, Q-DM, RepQ-ViT)
│   ├── 3.4.3 Outlier Processing (SmoothQuant, AWQ, QuaRot, QuIP#, SpinQuant, OmniQuant, OSTQuant)
│   └── 3.4.4 Rounding (AdaRound, FlexRound)
├── 3.5 Data Free Quantization (无数据量化)
│   ├── 3.5.1 Truly Data-Free (DFQ, SQuant)
│   ├── 3.5.2 Accurate and Diverse Generation (ZeroQ, GDFQ)
│   └── 3.5.3 Adversarial-Based Generation (ZAQ)
├── 3.6 Advanced Format (高级格式)
│   ├── 3.6.1 Float-Based (TF32, AdaptiveFloat, ANT, DFloat, MSFP)
│   ├── 3.6.2 Fixed Point (F8Net, Vs-quant)
│   └── 3.6.3 Other Formats (QLoRA/NF4, LoftQ, LUT-GEMM)
├── 3.7 Diffusion Model (扩散模型量化) → 详见 [[Diffusion Model Quantization]]
│   ├── 3.7.2 Calibration Set (PTQ4DM, Q-Diffusion, TFMQ-DM, APQ-DM)
│   ├── 3.7.3 Activation Distribution (Q-DM, TDQ, BiDM)
│   ├── 3.7.4 Quantization Error (PTQD, BitsFusion, MixDQ, SVDQuant, StepbaQ, MPQ-DM)
│   └── 3.7.5 Other DM Methods (ViDiT-Q, Q-DiT, PassionSR, DGQ)
└── 3.8 Other (其他)
```

### 与两步分解框架的映射

本综述的分类体系与 [[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 的两步分解框架并非竞争关系，而是互补视角：

| 本综述分类 | 两步分解框架对应 | 说明 |
|-----------|----------------|------|
| 3.4 Redistribution → Outlier Processing | Step 1：Scaling / Rotation | SmoothQuant、AWQ、QuaRot 等在本综述中归入 "Redistribution" |
| 3.4.4 Rounding (AdaRound) | Step 2：量化误差补偿 | 对应 GPTQ 类的逐权重优化 |
| 3.1 Better s and z | 贯穿 Step 1 和 Step 2 | 量化范围参数的优化 |
| 3.3 Mixed Precision | 正交维度 | 两步分解框架不涉及混合精度 |
| 3.7 Diffusion Model | 扩展到新架构 | 两步分解框架聚焦 LLM，未覆盖扩散模型 |

### 对本 wiki 的新贡献

1. **扩散模型量化**（3.7 节）——本 wiki 此前完全缺失此领域，现在通过 [[Diffusion Model Quantization]] 补充
2. **无数据量化**（3.5 节）——DFQ、ZeroQ 等方法完全不需要校准数据，与 PTQ 方法互补
3. **混合精度量化**（3.3 节）——HAWQ 系列、OWQ 等通过 Hessian 分析实现逐层/逐通道差异化精度分配
4. **高级格式**（3.6 节）——NF4（QLoRA）、AdaptiveFloat 等与 [[FP4 Quantization]] 相关的格式探索
5. **ViT 量化的独特挑战**——ViT 的激活和权重都沿 channel 维度呈现强相关性（Fig. 3），与 LLM 的 [[Emergent Outlier Features]] 现象相关但不完全相同

## 实验与发现

本综述以全景视角呈现，不提供统一的量化实验对比。但其分类体系揭示了几个重要趋势：

**趋势一：PTQ 与 QAT 的界限模糊化** —— 越来越多的 PTQ 方法使用 STE 来优化量化参数（如 [[OmniQuant]] 的 LWC/LET），而 QAT 方法也在追求更轻量的训练开销，二者正在融合。

**趋势二：从单一技术到组合优化** —— 最优方法往往是多种技术的叠加（Redistribution + Better s and z + Mixed Precision），这与 [[Quantization Insights|Insight 1]] 中 "互补组合" 的元原则一致。

**趋势三：模型架构专用量化** —— 从通用 CNN 量化到 ViT 专用（I-ViT、FQ-ViT）再到扩散模型专用（PTQ4DM、Q-Diffusion），量化方法越来越针对特定架构的特性设计。

**趋势四：硬件格式与量化的协同设计** —— TF32、MSFP、NF4 等格式的提出说明量化不仅是算法问题，还涉及硬件数据格式的创新。

## 局限性

- 综述截止时间约为 2024 年底，部分 2025 年最新工作（如 [[FlatQuant]] ICML 2025、[[OSTQuant]] 2025）未纳入
- 对 LLM 量化的两步分解统一框架的深度分析不如 [[A Comprehensive Evaluation on Quantization Techniques for LLMs]]
- 实验对比以定性分类为主，缺少统一 benchmark 下的定量对比

## 引用关系

← 覆盖方法：[[SmoothQuant]]、[[AWQ]]、[[GPTQ]]、[[QuaRot]]、[[QuIP#]]、[[SpinQuant]]、[[OmniQuant]]（均归入 3.4.3 Outlier Processing）
← 相关综述：[[A Comprehensive Evaluation on Quantization Techniques for LLMs]]（LLM 专用量化综述，两步分解框架）、[[Diffusion Model Quantization - A Review]]（浙大，扩散模型量化专门综述，提供了远更深入的 DM 量化分析和统一基准实验）
→ 新增覆盖：[[Diffusion Model Quantization]]（3.7 节，20+ 篇扩散模型量化论文的系统分类）
→ 配套资源：[Awesome-Model-Quantization](https://github.com/Kai-Liu001/Awesome-Model-Quantization)（持续更新的论文列表）
