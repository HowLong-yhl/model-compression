---
title: "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks"
aliases: [QuIP#, QuIP-sharp, QuIP Sharp]
tags: [model-compression, quantization, source, rotation, PTQ, weight-only, lattice, codebook]
created: 2026-04-20
updated: 2026-04-20
type: source-summary
venue: ICML 2024
authors: [Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa]
affiliation: Cornell University (RelaxML Lab)
url: https://arxiv.org/abs/2402.04396
code: https://github.com/Cornell-RelaxML/quip-sharp
---

## 一句话总结

QuIP# 将 Randomized Hadamard Transform（RHT）用于 incoherence processing，配合基于 **E₈ 格（lattice）** 的高效向量量化 codebook，在 2-bit 极端压缩下实现 SOTA，且首次证明 **3-bit 模型的 scaling 可以优于理论无损 4-bit 模型**。

## 核心问题

极低比特量化（≤4 bit）如何在保持模型质量的同时实现高效推理？前序方法（OmniQuant, GPTQ 等）在 2-3 bit 下性能崩塌。

## 方法三板斧

### 1. Randomized Hadamard Transform (RHT) — 改进 incoherence processing

**Incoherence** 的直觉：权重矩阵如果每个元素大小都差不多（没有 outlier），就更容易量化。数学上，µ-incoherent 矩阵的最大元素受 µ/√n 约束。

QuIP（前作）使用 Kronecker product 随机正交矩阵做 incoherence processing，复杂度 O(n√n)。QuIP# 改用 RHT：

```
x → V · S · x
```

其中 V 是 Hadamard 矩阵，S 是随机符号对角矩阵（元素 ∈ {±1}）。

**RHT 的三大优势**：
- 理论 incoherence bound 从 QuIP 的 log² 改善为 **log**（更紧）
- 计算复杂度从 O(n√n) 降至 **O(n log n)**（Fast Walsh-Hadamard Transform）
- 常数因子进一步减小（Hadamard 乘法无需浮点乘法，仅加减法）

**维度兼容**：当 n 不是 2 的幂时，分解 n = p·q（p 为最大 2 的幂），用 H_p ⊗ H_q 构造。若分解不可行，退化为 Randomized FFT（RFFT）。

### 2. E₈ Lattice Codebook — 最优向量量化

RHT 处理后的权重近似服从 **球形高斯分布**。此时标量量化（SQ）的超立方体覆盖效率低，需要球形 codebook 的向量量化（VQ）。

**E₈ 格**是 R⁸ 中的最优球堆积（Viazovska 2017，证明了 8 维 kissing number = 240），具有极高对称性。QuIP# 设计了 E8P codebook：

- 基于 D̂₈ 格（E₈ 的等价表示）
- 2 bit × 8 维 = 16 bit codeword
- 解码只需查找 **256 条目的表**（S 表）+ 7 bit 符号翻转 + 1 bit 偏移
- 总码本大小仅 **1 KiB**，完全可放入 GPU L1 cache

**Block LDLQ**：将原始 LDLQ 逐列自适应舍入推广到逐块（block-wise），支持 VQ。每次舍入一个 8 维向量而非单个标量，用线性反馈修正后续块的误差。

**3-bit / 4-bit 扩展**：使用 Residual VQ（RVQ）——多级码本级联：
- 4 bit = 2-bit E8P 码本 × 2（两次量化残差）
- 3 bit = 2-bit E8P + 1-bit E₈ 码本

### 3. Inter-layer Fine-tuning

量化后通过少量校准数据做层间微调，恢复量化误差：
- 先在 block 内微调未量化层以补偿已量化层
- 再跨 block 端到端微调
- 70B 模型仅需约 50 GPU-hours

## 实验亮点

| 模型 | 比特 | WikiText2 PPL | 关键发现 |
|------|------|---------------|---------|
| LLaMA-2 70B | 2-bit | 4.16 | 首次 2-bit 可用 |
| LLaMA-2 70B | 3-bit | — | **scaling 优于理论无损 4-bit** |
| LLaMA-2 7B | 2-bit | 8.22 | 大幅优于 QuIP (13.0) |
| 全系列 | 4-bit | — | 优于 OmniQuant, AQLM |

- 推理 CUDA kernel 实现达到 RTX 4090 峰值内存带宽的 **>50%**
- E8P 码本的 MSE 优于 D₄、K-means 等所有对比码本

## 理论贡献

**Theorem 4.1**（Block LDLQ 误差界）：给定 µ-incoherent Hessian H，使用满足 E[(Q(x)-x)(Q(x)-x)^T] ⪯ σ²I 的向量量化器 Q，Block LDLQ 的误差为：

```
E[tr((Ŵ-W)H(Ŵ-W)^T)] ≤ g·m·µ²·σ² / n · tr(H^{1/2})²
```

其中 g 为块大小。这比独立量化各块改善了 tr(H) → tr(H^{1/2})²/n 因子。

## 与其他方法的关系

- **vs [[GPTQ]]**：GPTQ 用标量 LDLQ（逐列自适应舍入），QuIP# 推广到 block VQ + 格码本
- **vs [[SpinQuant]]**：SpinQuant 优化旋转矩阵本身；QuIP# 使用固定 RHT 但在码本设计上做文章
- **vs [[AWQ]] / [[SmoothQuant]]**：这些是 scaling-based 方法，QuIP# 证明了 incoherence processing 是更强的 outlier 抑制手段
- **vs AQLM**：都用 VQ，但 AQLM 用非结构化 K-means codebook（慢），QuIP# 的 E₈ 结构化码本可快速推理
- **vs QuIP（前作）**：QuIP 用 Kronecker 正交矩阵 + 标量 LDLQ；QuIP# 改进为 RHT + Block LDLQ + E₈ + FT

## 关键 insights

1. **Incoherence 是量化的本质**：outlier suppression 的各种 heuristic（scaling, mixed-precision）本质上都在追求 incoherence，而 RHT 是数学上最优雅且有理论保证的方式
2. **VQ 的维度越高越好**：E₈（8 维）的 packing density 远优于 D₄（4 维），更远优于标量量化（1 维）
3. **3-bit > 理论无损 4-bit**：这打破了"4-bit 是最优比特率"的传统认知，表明 PTQ 技术本身还有巨大进步空间

## 局限性

- 仅支持 weight-only 量化，不量化激活
- E₈ 码本目前仅支持 2-bit 基础粒度（3/4-bit 通过 RVQ 扩展）
- 需要自定义 CUDA kernel 才能实现高效推理
- Fine-tuning 步骤增加了量化时间

## 引用关系

← 前作：QuIP (Chee et al., 2023)
← 理论基础：E₈ lattice (Viazovska 2017), LDLQ 最优性 (Chee et al. 2023)
← 团队：[[Cornell RelaxML Lab]]
→ 启发了：[[SpinQuant]]（将 Hadamard 旋转的 tighter bound 作为初始化依据）
→ 同时期：AQLM（VQ 路线的另一分支，非结构化码本）
