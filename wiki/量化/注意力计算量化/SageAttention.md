---
title: "SageAttention: Accurate 8-Bit Attention for Plug-and-Play Inference Acceleration"
aliases: [SageAttention, SageAttention1, Zhang 2024 SageAttention]
source_type: paper
source_url: "https://arxiv.org/abs/2410.02367"
source_date: 2024-10-03
ingested: 2026-04-29
venue: ICLR 2025
authors: [Jintao Zhang, Jia Wei, Haofeng Huang, Pengle Zhang, Jun Zhu, Jianfei Chen]
affiliation: Tsinghua University
code: "https://github.com/thu-ml/SageAttention"
tags: [model-compression, quantization, method, attention-kernel, inference, int8]
---

## 核心要点

SageAttention 是首个将量化系统性应用于 **attention 矩阵乘法本身**（而非权重或 KV cache）的 plug-and-play 加速方案。核心观察：现有量化研究几乎只关注线性层，但在长序列场景下 attention 的 O(N²) 计算才是主要瓶颈。

SageAttention 在 RTX4090 上达到 341 TOPS（FlashAttention2 的 ~2.1×），且几乎无端到端精度损失。

## 两个核心挑战与对策

### C1：K 矩阵的 Channel-wise Outlier

K 矩阵继承了 LLM 中 [[Emergent Outlier Features]] 的特征——特定通道持续产生极端值。直接 INT8 量化会导致灾难性精度损失。

**对策：Smooth K**。核心变换：$\gamma(K) = K - \text{mean}(K)$。由于 softmax 对常数偏移不变（$\text{softmax}(QK^T) = \text{softmax}(Q(K - \bar{K})^T)$），减去 token-wise 均值不改变 attention 输出，但消除了 channel-wise outlier。开销 <0.2%。

**效果**：Per-block 量化的 CosSim 从 30.60%（无 smoothing）提升到 99.31%。对比 FlashAttention3 的 FP8 量化仅 26.76% CosSim。

### C2：PV 矩阵乘法的精度

将 P（softmax 输出）和 V 都量化到 INT8 并不总能保证精度。

**对策：FP16 累加器**。不量化 P 和 V 到 INT8，而是保持 FP16 精度并使用 FP16 累加器。这在 RTX4090/3090 上反而更快（FP16 累加器比 FP32 累加器快 2×），同时精度与 FP32 累加器完全一致（CosSim 99.98%）。

## 方法详解

### 量化策略

| 矩阵 | 数据类型 | 量化粒度 | 说明 |
|------|---------|---------|------|
| Q | INT8 | per-token 或 per-block | INT8 比 FP8 在 4090/3090 上更快且更准 |
| K | INT8 | per-token 或 per-block | 先 smooth 再量化 |
| P | FP16 | — | 不量化，使用 FP16 累加器 |
| V | FP16 | — | 不量化 |

**为什么选 INT8 而非 FP8**：在 QK^T 矩阵乘法中，INT8 的 CosSim 为 99.54%，E4M3 仅 92.83%，E5M2 仅 77.95%。INT8 的均匀量化点对 smooth 后的均匀分布更合适——这与 [[Rotation-based Quantization]] 中"旋转后 INT4 优于 FP4"的发现一脉相承。

### 自适应量化

提供四种 kernel（SAGEAttn-T/B/vT/vB），根据每层的 CosSim 阈值（99.8%）自适应选择最快的 kernel。此策略额外带来 11.7% 加速。

### 融合优化

量化操作与 RoPE 层融合，1/√d 系数融入量化 scale，避免额外开销。

## 实验结果

### 速度

| 指标 | 数值 |
|------|------|
| 峰值吞吐 (RTX4090) | **341 TOPS**（理论 INT8 上限的 52%） |
| vs FlashAttention2 | **~2.1× 加速** |
| vs xformers | **~2.7× 加速** |

实际模型端到端加速：CogvideoX 2.01×, Llama2 1.77×, UltraPixel 2.14×, Unidiffuser 2.34×, TIMM 5.89×。

### 精度

| 模型 | 指标 | Full-Precision | SageAttention |
|------|------|---------------|---------------|
| Llama2-7B | WikiText PPL | 5.823 | 5.824 |
| TIMM | ImageNet Acc | 84.79% | 84.74% |
| Llava1.6 | TextVQA | 60.25% | 60.09% |

**对比 weight 量化**：SageAttention (PPL 5.4729) 精度远优于 AWQ W4A16 (PPL 5.5988)，说明 attention 量化比 weight 量化更友好。

## 与 Attention 量化/KV Cache 量化的区别

SageAttention 量化的是 **attention 计算过程**（Q·K^T 和 P·V 这两个矩阵乘法），不改变 KV cache 的存储精度。[[KV Cache Quantization]] 量化的是 **存储在 cache 中的 K/V 向量**。两者完全正交：可以先用 KIVI 2-bit 存储 KV cache，在计算 attention 时用 SageAttention INT8 加速矩阵乘——同时压缩存储和计算。

## 引用关系

← 理论基础：[[Emergent Outlier Features]]（K 的 channel-wise outlier），FlashAttention（tiling + online softmax 框架）
← 技术关联：[[FP4 Quantization]]（INT8 vs FP8 的比较），[[Rotation-based Quantization]]（smooth 后 INT 优于 FP 的发现）
→ 后续工作：[[SageAttention2]]（INT4 + FP8），[[SageAttention3]]（FP4 microscaling）
→ 正交互补：[[KV Cache Quantization]]（存储量化），[[Token Eviction]]（cache 压缩）
→ 概念关联：[[Quantized Attention Kernel]]
