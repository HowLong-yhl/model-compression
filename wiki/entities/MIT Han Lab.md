---
title: MIT Han Lab
aliases: [Han Lab, MIT HAN Lab, Song Han Lab]
tags: [model-compression, entity, lab]
created: 2026-04-20
updated: 2026-05-08
sources: [SmoothQuant, AWQ, SVDQuant, StreamingLLM, QServe]
---

## 概述

MIT Han Lab 是由宋涵（Song Han）教授领导的、位于 MIT EECS 的研究实验室，是 LLM 高效推理和模型压缩领域最具影响力的团队之一。在本 wiki 收录的多篇经典论文中，有多篇出自该实验室。

## 与本 Wiki 的关联

### 核心贡献

- **[[SmoothQuant]]**（ICML 2023）：第一作者 Guangxuan Xiao，提出等价变换平滑激活 outlier 的 W8A8 PTQ 方案
- **[[AWQ]]**（MLSys 2024 Best Paper）：第一作者 Ji Lin，提出激活感知的权重缩放保护 salient weights
- **[[StreamingLLM]]**（ICLR 2024）：第一作者 Guangxuan Xiao，发现 [[Attention Sink]] 现象——初始 token 始终获得高 attention weight，据此提出 sink + sliding window 方案实现无限长上下文推理。该工作是 [[Token Eviction]] 领域的奠基性发现之一
- **[[QServe]]**（MLSys 2025）：第一作者 Yujun Lin、Haotian Tang，提出 QoQ W4A8KV4 渐进式量化算法 + GPU serving 系统协同设计，在 L40S 上实现 1.5-3.5× 吞吐提升
- **SVDQuant**（ICLR 2025）：用低秩分支（SVD）吸收权重 outlier 实现扩散模型 W4A4 量化，配套 Nunchaku 推理引擎（详见 [[Diffusion Model Quantization]]）

五项工作展现了 Han Lab 从 LLM 量化（SmoothQuant/AWQ）到量化感知服务（QServe）到高效推理（StreamingLLM）再到生成模型压缩（SVDQuant）的研究脉络。SmoothQuant、AWQ 和 QServe 共享"等价缩放变换"的技术基因（见 [[Equivalent Scaling Transformation]]），StreamingLLM 则开辟了 KV cache 管理的新方向（见 [[KV Cache Management]]）。

### 成员交叉

- **Ji Lin**：AWQ 第一作者，SmoothQuant 共同作者
- **Yujun Lin**：QServe 共同第一作者
- **Haotian Tang**：QServe 共同第一作者、系统设计主导
- **Guangxuan Xiao**：SmoothQuant 第一作者，AWQ/QServe 共同作者
- **Song Han**：所有论文的通讯作者

### TinyChat / QServe / Nunchaku

AWQ 论文同时推出了 TinyChat 推理框架，实现 4-bit 模型在桌面和边缘 GPU 上的高效部署，加速比达 3.5-3.7×。[[QServe]] 推出了同名 serving 系统（后更名 OmniServe），面向数据中心高并发 LLM 推理。SVDQuant 则推出了 Nunchaku 引擎，专为 W4A4 扩散模型推理设计，FLUX.1-12B 在 4090 上实现 3.5× 加速。

## 工业影响

SmoothQuant 已被 NVIDIA TensorRT-LLM、Microsoft ONNX Runtime、Amazon SageMaker、Intel Neural Compressor 等主流推理框架采纳，成为 W8A8 量化的事实标准。

## 链接

- 实验室主页：https://hanlab.mit.edu/
- SmoothQuant: https://github.com/mit-han-lab/smoothquant
- AWQ: https://github.com/mit-han-lab/llm-awq
