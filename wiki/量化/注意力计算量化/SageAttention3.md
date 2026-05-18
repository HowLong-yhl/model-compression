---
title: "SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-bit Training"
aliases: [SageAttention3, SageAttn3, Zhang 2025 SageAttention3]
source_type: paper
source_url: "https://arxiv.org/abs/2505.11594"
source_date: 2025-05-17
ingested: 2026-04-29
venue: NeurIPS 2025
authors: [Jintao Zhang, Jia Wei, Haoxu Wang, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Kai Jiang, Jianfei Chen, Jun Zhu]
affiliation: Tsinghua University, ShengshuTech
code: "https://github.com/thu-ml/SageAttention"
tags: [model-compression, quantization, method, attention-kernel, inference, training, fp4, int8]
---

## 核心要点

SageAttention3 有两大贡献：

1. **首个 FP4 Attention**：利用 Blackwell GPU 的 FP4 Tensor Core 加速推理，在 RTX5090 上达到 **1038 TOPS**（FlashAttention2 的 **5×**）
2. **首个可训练低比特 Attention（SageBwd）**：将 8-bit 量化扩展到 attention 的反向传播，在 fine-tuning 任务上**无损**，但 pretraining 收敛较慢

## FP4 Attention（推理加速）

### Microscaling NVFP4 量化

Blackwell GPU 的 FP4 matmul 吞吐约 1600 TOPS（FP16 的 8×）。SageAttention3 对 QK^T 和 PV 都使用 NVFP4 microscaling 量化：

- **NVFP4 格式**：E2M1，block size 1×16，scale 用 E4M3 FP8 表示
- **为什么选 NVFP4 而非 MXFP4**：NVFP4 CosSim 99.52% vs MXFP4 98.37%（MXFP4 的 block size 1×32 太大，E8M0 scale 精度不够）

这里 NVFP4 的 microscaling 粒度（1×16）本质上与 [[FP4 Quantization]] 中讨论的 Blackwell 原生 FP4 格式一致。

### Two-level Scaling for P

P（softmax 输出）的值在 [0, 1] 范围内，直接用 NVFP4 microscaling 量化时 scale factor 被压缩到 [0, 0.167]，E4M3 的表示能力严重浪费。

**对策**：Two-level quantization——先 per-token 归一化到 [0, 448×6]，再做标准 FP4 microscaling。CosSim 从 93.32%（直接量化）提升到 **99.52%**。

### 硬件优化

三项 kernel 优化使实际吞吐接近理论上限：

1. **Permutation for K**：重排 P tile 的列布局以适配 FP4 累加器的寄存器 layout
2. **Reuse Shuffle**：量化与 online softmax 融合，减少 50% 冗余 shuffle 操作，~10% 全 kernel 加速
3. **Producer Warp Epilogue**：ping-pong 调度替代传统 consumer-based 方案

### 理论吞吐对比

| 方法 | B300 | B200 | RTX5090 |
|------|------|------|---------|
| FlashAttention3 (FP16) | 2500 TOPS | 2500 TOPS | 209.5 TOPS |
| FlashAttention3 (FP8) | 5000 TOPS | 5000 TOPS | 419 TOPS |
| **SageAttention3 (FP4)** | **15000 TOPS** | **10000 TOPS** | **1676 TOPS** |

## SageBwd（可训练 8-bit Attention）

### 核心发现：dO·V^T 是精度瓶颈

Attention 反向传播有 5 个 matmul。实验发现 **dO·V^T** 的量化误差会通过 dP → dS → dQ/dK 沿序列长度方向持续累积。保持 dO·V^T 在 FP16 精度，其余 4 个 matmul 用 INT8 per-block 量化即可。

| 矩阵乘法 | 精度 | 说明 |
|---------|------|------|
| QK^T | INT8 | 前向，smooth K + per-block |
| PV | INT8 | 前向，per-token P + per-block V |
| P^T·dO → dV | INT8 | 反向 |
| **dO·V^T → dP** | **FP16** | **反向，精度瓶颈** |
| dS·K → dQ, dS^T·Q → dK | INT8 | 反向 |

### INT8 vs FP8

| 指标 | INT8 SageBwd | FP8 SageBwd |
|------|-------------|-------------|
| dQ L1 误差 | 0.0290 | 0.0696 |
| dQ CosSim | 0.9987 | 0.9880 |
| 硬件支持 | A100, MI250, Ascend 910B 等 | 仅 H100, RTX 40xx |

选择 INT8 因为精度更高且硬件覆盖面更广。

### Fine-tuning 结果

| 模型 | 方法 | GSM8K | MMLU | HellaSwag |
|------|------|-------|------|-----------|
| Qwen2.5-1.5B | BF16 | 0.521 | 0.569 | 0.905 |
| Qwen2.5-1.5B | **SageBwd** | **0.520** | **0.574** | **0.911** |
| Qwen2.5-3B | BF16 | 0.601 | 0.640 | 0.944 |
| Qwen2.5-3B | **SageBwd** | **0.607** | **0.653** | **0.943** |

Fine-tuning 完全无损，甚至在部分指标上超过 BF16。

### INT8 Fine-tuning + FP4 推理的协同

INT8 SageBwd 微调后用 FP4 SageAttention3 推理，精度**高于** BF16 微调 + FP4 推理：Qwen2.5-1.5B GSM8K 0.5232 vs 0.4912。推测原因是 INT8 和 FP4 的量化分布更相似，减少了训练-推理的 mismatch。

### Pretraining 的局限

在 FineWeb-Edu 上 pretraining Llama-400M 时，SageBwd 可以收敛但速度较慢——目前不适用于 pretraining，这是开放问题。

## 实验结果

### 推理速度

| 模型 | GPU | FlashAttn2 | SageAttn3 | 加速比 |
|------|-----|-----------|-----------|--------|
| CogvideoX-2B | RTX5090 | 64s | **27s** | **2.4×** |
| HunyuanVideo | RTX5090 | 489s | **164s** | **3.0×** |

### Speed-Accuracy 全系列对比

| 方法 | TOPS (5090) | CosSim |
|------|------------|--------|
| FlashAttention2 | 214 | 100.000% |
| SageAttention1 | 479 | 99.996% |
| SageAttention2-8b | 643 | 99.995% |
| FlashAttention3-FP8 | — (H100: 890) | 98.570% |
| **SageAttention3** | **1038** | **99.551%** |

SageAttention3 在 5× 加速的同时精度远超 FlashAttention3-FP8。

## 引用关系

← 前序工作：[[SageAttention]]（INT8 框架），[[SageAttention2]]（INT4 + FP8 + smooth Q/K + per-thread 量化）
← 技术关联：[[FP4 Quantization]]（NVFP4 microscaling 格式），FlashAttention（tiling + online softmax）
→ 贡献：首个 FP4 attention kernel，首个可训练低比特 attention
→ 正交互补：[[KV Cache Quantization]]（存储量化），[[Token Eviction]]（cache 压缩）
→ 概念关联：[[Quantized Attention Kernel]]
