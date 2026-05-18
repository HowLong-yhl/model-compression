---
title: Quantization Granularity
aliases: [量化粒度, Granularity, Per-tensor, Per-channel, Per-group]
tags: [model-compression, quantization, concept, setting]
created: 2026-04-20
updated: 2026-04-20
sources: [A Comprehensive Evaluation on Quantization Techniques for LLMs]
---

## 定义

Quantization Granularity（量化粒度）指在量化过程中，共享同一组量化参数（scale Δ 和 zero-point Z）的数据范围大小。粒度越细，每组数据越少，量化越精确，但存储 overhead 越大。

## 三种主要粒度

### Per-tensor

整个 tensor 使用一组 (Δ, Z)。最粗粒度，overhead 最小，但在 LLM 中因 [[Emergent Outlier Features]] 导致精度极差。

### Per-channel / Per-token

- **Per-channel**（用于权重）：每个输入通道独立量化
- **Per-token**（用于激活）：每个 token 独立量化

这是早期 W8A8 方法（如 [[SmoothQuant]]、[[LLM.int8()]]）的常用粒度。

### Per-group

将通道/token 内的元素进一步分成固定大小的组（如 group size = 128、64、32），每组独立量化。这是低比特量化（W4A4）的主流选择。

## 精度 vs 存储的 Trade-off

来自 [[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 的实验（LLaMA-3.2-1B, Rotation+Scaling+GPTQ）：

| Group Size | 额外 bits/参数 | WikiText-2 PPL |
|-----------|---------------|---------------|
| 32 | 1.0 | 最好 |
| 64 | 0.5 | ↓ |
| 128 | 0.25 | ↓ |
| 256 | 0.125 | ↓ |
| 512 | 0.0625 | 最差 |

group size 越小，额外存储的 FP32 scaling factors 带来的 bit overhead 越大。Group-128 是实践中的常用平衡点。

## 各方法的粒度选择

| 方法 | 权重粒度 | 激活粒度 |
|------|---------|---------|
| [[SmoothQuant]] | Per-tensor | Per-tensor/Per-channel/Per-token |
| [[GPTQ]] | Per-group | — (weight-only) |
| [[AWQ]] | Per-group | — (weight-only) |
| [[LLM.int8()]] | Per-channel | Per-token (+ FP16 分解) |
| QuaRot | Per-channel/Per-group | Per-channel |
| SpinQuant | Per-channel | Per-channel |
| MXFP4 | **Per-group (g=32, 固定)** | Per-group (g=32, 固定) |
| NVFP4 | **Per-group (g=16, 固定)** | Per-group (g=16, 固定) |

## 演进趋势

随着 [[Rotation-based Quantization]] 的引入（QuaRot、SpinQuant），rotation 大幅消除 outlier 后，**per-channel 量化的性能被显著提升**，接近 per-group 的效果。这使得后续方法（FlatQuant、OSTQuant、ResQ）也转向 per-channel 粒度，在性能和 overhead 间取得更好平衡。

FP4 格式（MXFP4、NVFP4）采用**固定小 group size**（16-32），其 per-group 量化由硬件加速，不构成 overhead 瓶颈。

## 与 FP4 的交互

细粒度 group 对 FP4 和 INT4 的影响截然不同：
- **粗粒度（per-channel）**：FP4 远优于 INT4（PPL 135 vs 257），因为 FP4 的非均匀量化点更适配正态分布
- **细粒度（group-16~32）**：FP4 与 INT4 差距大幅缩小，因为局部分布趋于均匀

这解释了为什么 NVFP4 使用 group-16：更细的 group 能弥补 FP4 有限位宽的不足。详见 [[FP4 Quantization]] 和 [[Quantization Insights|Insight 6]]。

