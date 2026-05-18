---
title: "SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization"
aliases: [SageAttention2, SageAttn2, Zhang 2024 SageAttention2]
source_type: paper
source_url: "https://arxiv.org/abs/2411.10958"
source_date: 2024-11-17
ingested: 2026-04-29
venue: ICML 2025
authors: [Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, Jianfei Chen]
affiliation: Tsinghua University
code: "https://github.com/thu-ml/SageAttention"
tags: [model-compression, quantization, method, attention-kernel, inference, int4, fp8]
---

## 核心要点

SageAttention2 在 [[SageAttention]] 基础上进一步将 QK^T 量化从 INT8 降到 **INT4**，PV 量化到 **FP8**。两个关键改进：(1) 提出 smooth Q 进一步提升 INT4 QK^T 精度；(2) 发现 FP8 tensor core 的累加器实际上只有 **FP22**（非 FP32），提出 two-level accumulation 策略。

峰值吞吐 481 TOPS（RTX4090），比 FlashAttention2 快 **~3×**，比 xformers 快 **~4.5×**。在 Hopper GPU 上与 FlashAttention3(FP8) 速度持平，但精度远优。

## 解决 SageAttention 的两个弱点

### W1：INT8 仅为 INT4 速度的一半

SageAttention 用 INT8 量化 QK^T，但 INT4 matmul 的吞吐是 INT8 的 2×。然而 INT4 的数值范围极小（仅 15 个可表示值），直接量化精度崩溃。

### W2：FP16 累加器仅在部分 GPU 有加速

SageAttention 的 FP16 累加器策略仅在 RTX 4090/3090 上有速度优势，在 A100、L40、H100 等其他 GPU 上无加速。

## 核心技术

### Smooth Q

SageAttention 仅 smooth K（减去 token-wise 均值）。SageAttention2 进一步 smooth Q：

$$\gamma(Q_i) = Q_i - \bar{q}_i, \quad \gamma(K_j) = K_j - \bar{k}$$

attention score 分解为：$S_{ij} = \gamma(Q_i)\gamma(K_j)^T + \Delta S_{ij} + b$

其中 $\Delta S_{ij}$ 是可以用低成本 GEMV 精确计算的修正项。同时 smooth Q 和 K 的 CosSim 从 80.04%（无 smoothing）提升到 **99.46%**。

**Smoothing 效果排序**：Q+K (99.46%) > Q-only (98.30%) > K-only (98.07%) > Hadamard (79.77%) > SmoothAttn (90.21%) > 无 (80.04%)

### Per-thread INT4 量化

提出硬件友好的 per-thread 量化粒度——利用 GPU warp 内线程的自然分组，每个线程独立量化其负责的数据切片。

| 粒度 | CosSim | TOPS |
|------|--------|------|
| Per-tensor | 97.15% | 286 |
| Per-block | 98.03% | 284 |
| **Per-thread** | **99.45%** | **283** |
| Per-token | 99.45% | 268 |

Per-thread 精度等同于 per-token（精细量化组，32× finer than per-block），但速度接近 per-block（无额外 overhead）。

### FP8 PV + Two-level Accumulation

P 和 V 量化到 FP8 E4M3。但发现 Tensor Core 的 `mma.f32.f8.f8.f32` 指令的累加器实际只有 **FP22 精度**（1 sign + 8 exponent + 13 mantissa），导致精度损失。

**对策**：Two-level accumulation——用真正的 FP32 buffer 定期从 FP22 累加器中收集结果，0% 额外开销。

### 数据类型选择

| Q,K 类型 | P,V 类型 | CosSim | 说明 |
|---------|---------|--------|------|
| INT4 | INT8 | 77.05% | INT8 对 P 的 [0,1] 分布不友好 |
| INT4 | E5M2 | 99.20% | 可接受 |
| INT4 | **E4M3** | **99.44%** | 最优 |
| INT4 | FP16 | 99.45% | 参考上限 |

## 实验结果

### 速度

| GPU | vs FlashAttention2 | vs xformers |
|-----|-------------------|-------------|
| RTX4090 | **2.93×** | **~4.5×** |
| L20 | **2.46×** | — |
| H100 | **2.61×** | — |
| H20 | **3.12×** | — |

端到端加速：CogvideoX(1.5-5B) 1040s→577s (1.8×), HunyuanVideo 2221s→1486s, Mochi 2336s→1316s。

### 精度

| 模型 | 指标 | Full-Precision | SageAttn2-8b | SageAttn2-4b |
|------|------|---------------|-------------|-------------|
| Llama3.1 | WikiText PPL | 6.013 | 6.019 | 6.256 |
| Llama3.1 | MMLU | 0.635 | 0.634 | 0.607 |

**对比 FlashAttention3(FP8)**：在 CogvideoX-1.5-5B 上 FlashAttn3-FP8 的 VQA-a 从 70.231 崩溃到 6.531，而 SageAttn2-8b 保持 69.492。在 Llama-3-262k InfiniBench 上 SageAttn2 平均分 43.06 vs FlashAttn3 41.53（因 Retr.KV 从 7.0 降到 0.4）。

## 技术开销

| 技术 | 速度开销 |
|------|---------|
| Per-thread 量化 | 0.35% |
| Smooth Q | 3.7% |
| Two-level accumulation | **0%** |

## 引用关系

← 前序工作：[[SageAttention]]（INT8 attention 基础框架，smooth K）
← 技术关联：[[Emergent Outlier Features]]（Q/K 的 outlier 需要 smoothing），[[FP4 Quantization]]（INT4 vs FP4 的比较）
→ 后续工作：[[SageAttention3]]（FP4 microscaling，Blackwell GPU）
→ 正交互补：[[KV Cache Quantization]]，[[Token Eviction]]
→ 概念关联：[[Quantized Attention Kernel]]
