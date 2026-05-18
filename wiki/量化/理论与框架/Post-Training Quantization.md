---
title: Post-Training Quantization
aliases: [PTQ, 训练后量化]
tags: [model-compression, quantization, concept, post-training]
created: 2026-04-20
updated: 2026-04-20
sources: [LLM.int8(), GPTQ, SmoothQuant, AWQ, A Comprehensive Evaluation on Quantization Techniques for LLMs]
updated: 2026-04-20
---

## 定义

Post-Training Quantization (PTQ) 是指在模型训练完成后、不进行重新训练的情况下，将模型参数（权重和/或激活）从高精度（FP32/FP16）映射到低精度（INT8/INT4 等）的技术。与 Quantization-Aware Training (QAT) 不同，PTQ 不需要访问完整训练数据集或训练流程，仅需少量 calibration 数据。

## 核心优势

PTQ 之所以成为 LLM 量化的主流范式，原因在于：大模型训练成本极高（数百万美元级别），重新训练不现实。PTQ 允许在训练完成后、部署前的"最后一公里"进行压缩，calibration 通常只需几十到几百个样本。

## 两大分支

### [[Weight-Only Quantization]]（权重量化）

仅量化权重，激活保持 FP16。代表方法：

- **[[GPTQ]]**（ICLR 2023）：基于二阶信息（Hessian 逆矩阵）的 one-shot 权重量化，可将 175B 模型压到 3-4 bit
- **[[AWQ]]**（MLSys 2024）：基于激活感知的等价缩放保护 salient weights，calibration 鲁棒性更强
- **RTN**（Round-to-Nearest）：最简单的基线，直接四舍五入

典型格式：W4A16（4-bit 权重，FP16 激活）

### Weight-Activation Quantization（权重+激活量化）

同时量化权重和激活。代表方法：

- **[[LLM.int8()]]**（NeurIPS 2022）：vector-wise 量化 + mixed-precision decomposition 处理 outlier
- **[[SmoothQuant]]**（ICML 2023）：通过等价变换平滑激活 outlier，实现高效 W8A8

典型格式：W8A8（8-bit 权重，8-bit 激活）

### W4A4 Quantization（权重+激活 4-bit 量化）

当前研究的主要前沿。同时将权重和激活量化到 4-bit，对内存效率和推理速度的提升潜力最大，但挑战也最大。代表方法通常组合使用 [[Rotation-based Quantization]] + [[Equivalent Scaling Transformation]] + [[GPTQ]] + Low-rank compensation。根据 [[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 的实验，最优组合在 LLaMA-3.2-1B 上可将 PPL 从 BF16 的 9.76 控制到 11.73。

此外，[[FP4 Quantization]]（MXFP4、NVFP4）作为 INT4 的替代数据格式，在 NVIDIA Blackwell 架构上获得硬件原生支持，是 W4A4 的新方向。

## 两步分解框架

[[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 提出了一个统一分类视角，将所有 PTQ 方法拆解为两步：

1. **预量化变换**（Pre-quantization Transformation）：Shifting、[[Equivalent Scaling Transformation|Scaling]]、[[Rotation-based Quantization|Rotation]]
2. **量化误差补偿**（Quantization Error Mitigation）：RTN、GPTQ（self-compensation）、Low-rank compensation

详见该来源摘要页。

## 共性挑战

所有 PTQ 方法都面临 [[Emergent Outlier Features]] 带来的挑战——在 ~6.7B+ 参数模型中，少数激活通道的幅度远超正常范围，使得传统量化方案崩溃。四篇经典论文各自提出了不同的应对策略：

| 方法 | 应对策略 | 量化类型 |
|------|---------|---------|
| [[LLM.int8()]] | 运行时 FP16/INT8 混合分解 | W8A8 |
| [[SmoothQuant]] | 离线等价变换平滑激活 | W8A8 |
| [[GPTQ]] | Hessian-based 误差补偿 | Weight-only |
| [[AWQ]] | 激活感知缩放保护 salient weights | Weight-only |

## Calibration

PTQ 方法通常需要少量 calibration 数据来收集统计信息：

| 方法 | Calibration 数据量 | 数据集 |
|------|-------------------|--------|
| [[LLM.int8()]] | 极少（运行时动态） | 无需离线 calibration |
| [[SmoothQuant]] | 512 sequences | The Pile |
| [[GPTQ]] | 128 segments × 2048 tokens | C4 |
| [[AWQ]] | 16 sequences | The Pile |

## 量化设置的关键选择

### 对称 vs 非对称量化

[[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 的实验揭示了一个重要且违反直觉的结论：**激活非对称化极其重要，权重非对称化几乎无用**。

| 设置 | PPL (↓) | 改善 |
|------|---------|------|
| 完全对称 | 12.73 | 基线 |
| 仅权重非对称 | 12.70 | 微乎其微 |
| **仅激活非对称** | **11.80** | **显著** |
| 全部非对称 | 11.70 | 额外微小收益 |

原因：激活分布天然非对称（经 SiLU/ReLU 后有非零均值），对称量化浪费一半范围；权重分布天然对称（均值~0），对称量化已最优。

**实用建议**：权重对称 + 激活非对称 = 最佳工程折中。详见 [[Quantization Insights|Insight 4]]。

## 部署格式

PTQ 方法落地为具体部署格式时，需考虑推理硬件、引擎兼容性、量化速度等工程因素。GPTQ 和 AWQ 是 GPU 推理（vLLM/TGI）的主流选择；GGUF 覆盖 CPU/Mac/混合推理；FP8 面向数据中心 Hopper+ GPU；BitsAndBytes 适合快速实验和 QLoRA 微调。详见 [[Quantization Deployment Formats]] 和 [[Model Quantization Ecosystem]]。
