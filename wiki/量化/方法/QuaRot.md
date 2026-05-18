---
title: "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs"
aliases: [QuaRot]
tags: [model-compression, quantization, source, rotation, PTQ, unoptimized-rotation, W4A4]
created: 2026-04-20
updated: 2026-04-20
type: source-summary
venue: NeurIPS 2024
authors: [Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, James Hensman]
affiliation: ETH Zurich / EPFL / IST Austria / Microsoft Research
url: https://arxiv.org/abs/2404.00456
code: https://github.com/sashkboos/QuaRot
---

## 一句话总结

QuaRot 利用 **Computational Invariance** 定理将随机 Hadamard 矩阵融入 Transformer 权重，使所有权重、激活和 KV cache 均无 outlier，首次实现端到端 INT4 全量化（W4A4KV4），且旋转步骤**零 calibration 数据、零优化**。

## 核心问题

如何在**不需要学习或校准**的前提下消除 LLM 中的 outlier，使得简单的 per-tensor INT4 量化即可工作？

## 方法详解

### Computational Invariance 定理

核心定理（源自 SliceGPT）：在 pre-norm Transformer 中，可以对 block 间的激活施加正交变换 Q 而不改变模型输出，因为 RMSNorm 满足交换性质：

```
RMSNorm(XQ⊤) · Q = RMSNorm(X)
```

这意味着可以自由选择 Q 来优化量化友好性，而**不需要任何重训练或数据**。

### 两阶段方法

**Stage 1: 权重修改（离线、无数据）**

1. **Stage 1a — 全局旋转**：选取随机 Hadamard 矩阵 Q，将 RMSNorm 的 diag(α) 融入权重后，对所有输入权重左乘 Q⊤：`W_k ← Q⊤·diag(α)·W_k`，输出权重右乘 Q：`W_down ← W_down·Q`。激活变为 `X ← XQ`，outlier 被均匀分散。

2. **Stage 1b — FFN 在线旋转**：在 FFN down-projection 前插入 Hadamard 变换 H，将 H 的逆融入 W_down。目的：消除 gate·up 乘积后产生的新 outlier。

3. **Stage 1c — Attention 在线旋转**：在 V projection 后插入 Hadamard，消除 KV cache 中的 outlier，使 KV cache 也可 4-bit 量化。

**Stage 2: 量化（可选 GPTQ）**

- 权重：默认用 GPTQ（可选 RTN）量化为 INT4
- 激活：在线 Round-to-Nearest 量化为 INT4
- KV cache：在线 RTN 量化为 INT4

### 关键设计选择

| 决策 | 选择 | 理由 |
|------|------|------|
| 旋转矩阵类型 | 随机 Hadamard | O(n log n) 快速变换，无浮点乘法 |
| 是否优化旋转 | **否** | 零数据依赖，纯代数操作 |
| 在线 Hadamard 位置 | FFN + Attention | 覆盖所有 outlier 产生点 |
| 权重量化器 | GPTQ | 提供 Hessian-based 误差补偿 |

## 实验亮点

| 模型 | 设定 | WikiText-2 PPL | vs FP16 Δ | Speedup |
|------|------|---------------|-----------|---------|
| LLaMA-2 70B | W4A4KV4 | +0.47 | 极小退化 | 3.33× prefill |
| LLaMA-2 70B | W6A6KV6 | 无损 | 0 | — |
| LLaMA-2 70B | W8A8KV8 | 无损 | 0 | — |

- Zero-shot 准确率保持 99%（70B 模型）
- 解码内存节省 **3.89×**
- 6-bit/8-bit 下用纯 RTN（无 GPTQ）即可无损

## 与其他方法的关系

- **vs [[SpinQuant]]**：QuaRot 用随机旋转（不优化），SpinQuant 学习旋转。SpinQuant 在 LLaMA-3 8B 上比 QuaRot 改善 45.1%，但 QuaRot 完全不需要校准数据
- **vs [[QuIP#]]**：两者都用 Hadamard 旋转做 incoherence processing。QuIP# 专注 weight-only + lattice VQ；QuaRot 覆盖 W+A+KV 全量化
- **vs [[SmoothQuant]] / [[AWQ]]**：Scaling 方法在 W4A4 下效果有限；QuaRot 的旋转更彻底地消除 outlier
- **vs SliceGPT**：两者共享 Computational Invariance 定理，但 SliceGPT 用于剪枝，QuaRot 用于量化

## 局限性

- 随机旋转引入性能方差（SpinQuant 发现最高 13 分差距）
- 在线 Hadamard 变换增加推理延迟（约 8%）
- 小模型（<7B）上 W4A4 退化仍较大
- 不做任何 calibration-based 优化，留下了进一步提升的空间

## 引用关系

← 理论基础：Computational Invariance (SliceGPT, Ashkboos et al. 2024), Incoherence (QuIP/QuIP#)
← 分类：[[Unoptimized Rotation]]（使用固定随机 Hadamard，不优化旋转矩阵本身）
→ 启发了：[[SpinQuant]]（在 QuaRot 基础上学习旋转）、OSTQuant、FlatQuant
→ KV cache 量化：QuaRot 的 W4A4**KV4** 方案本质上也做了 KV cache 量化——通过 Stage 1c 的 Attention 在线旋转消除 KV cache 中的 outlier 后用 INT4 量化。后续 [[KVQuant]]、[[KIVI]] 等专门的 KV cache 量化方法进一步细化了 Key/Value 的不对称处理。详见 [[KV Cache Quantization]]
→ 相关：[[Quantization Insights]]（验证了旋转的有效性与局限性）
