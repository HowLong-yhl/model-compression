---
title: Lattice Codebooks (E₈ Vector Quantization)
aliases: [E8 Lattice, E8P Codebook, Lattice VQ, 格码本]
tags: [model-compression, quantization, concept, codebook, vector-quantization]
created: 2026-04-20
updated: 2026-04-20
sources: [QuIP#]
---

## 定义

Lattice Codebooks 是 [[QuIP#]] 引入的基于数学格（lattice）结构的向量量化码本，核心是 **E₈ 格**——8 维空间中的最密球堆积。与传统标量量化（每次量化一个数值）不同，lattice VQ 将 8 个权重作为一个向量联合量化到格点上，利用格的高对称性实现高效解码。

## 从标量量化到向量量化

### 标量量化（SQ）的局限

传统 [[Post-Training Quantization|PTQ]] 逐个权重量化（如 [[GPTQ]] 的 LDLQ），可表示的值集合形成高维超立方体。但经过 [[Rotation-based Quantization|旋转 incoherence processing]] 后，权重分布变为近似**球形高斯**——超立方体覆盖球形分布的效率很低。

### 向量量化（VQ）的优势

VQ 将 d 个权重联合量化到 d 维码本 C 中的码字。k-bit VQ 的码本大小为 2^(kd) × d。维度 d 越大，码本形状越能匹配源分布，quantization distortion 越低。但代价是码本大小指数增长，高维/高比特率时不可行。

### 格码本的解法

格（lattice）是 R^d 中的离散周期结构，具有高度对称性。利用对称性，可以用极少存储实现高效编解码，绕过指数爆炸问题。

## E₈ 格

E₈ 是 8 维欧几里得空间中的一个特殊格，定义为：

```
E₈ = (Z⁸ ∪ (Z⁸ + 1/2)) ∩ { x | 1ᵀx 是偶数 }
```

即所有坐标为整数或所有坐标为半整数的 8 维向量，满足坐标之和为偶数。

### 关键数学性质

- **最优球堆积**：E₈ 在 8 维中达到最密球堆积（Viazovska 2017，解决了自 1900 年以来的 Hilbert 第 18 问题的 8 维情形）
- **Kissing number = 240**：每个格点恰有 240 个最近邻
- **高度对称**：对称群 |Aut(E₈)| = 696729600
- **自对偶**：E₈ 是自对偶格，意味着其 Voronoi cell 也是最优的

## E8P 码本设计

QuIP# 设计了 E8P（E8 Padded）码本，实现 **2-bit / 8-dimension** 的高效量化：

### 构造步骤

1. 利用 D̂₈ 格（E₈ 的等价表示）的对称性：翻转偶数个符号仍在格中
2. 选取 |D̂₈| 中范数 ≤ √10 的 227 个绝对值模式作为源码本 S，加 29 个范数 √12 的 padding 条目，共 256 = 2⁸ 个条目
3. 每个 16-bit codeword 编码为：8 bit 索引 S + 7 bit 符号翻转 + 1 bit ±1/4 偏移
4. 解码仅需查 **256 条目**的表 + 位运算

### 硬件友好性

- 整个码本仅 **1 KiB**
- 可完全放入 GPU L1 cache（即使 32× bank conflict 复制后仍可）
- 解码只需整数运算（查表+符号翻转），无浮点除法
- CUDA 实现达到 RTX 4090 峰值带宽 >50%

## 扩展到更高比特率：Residual VQ

E₈ 原生支持 2-bit。更高比特率通过 **Residual Vector Quantization (RVQ)** 实现：

| 目标比特 | 方案 | 说明 |
|---------|------|------|
| 2-bit | E8P × 1 | 基础方案 |
| 3-bit | E8P (2-bit) + E₈ (1-bit) | 两级残差 |
| 4-bit | E8P × 2 | 量化残差再量化 |

RVQ 公式：`RVQ(x, p, q) = Σᵢ δᵢ`，其中 `δᵢ = Q_qᵢ(x - Σⱼ<ᵢ δⱼ) / sᵢ · sᵢ`

## 理论比较

| 码本类型 | 维度 | 球堆积密度 | 推理速度 | 适用比特率 |
|---------|------|-----------|---------|-----------|
| Scalar (uniform grid) | 1 | 低 | 极快 | 任意 |
| D₄ lattice | 4 | 中 | 快 | 2-4 bit |
| **E₈ lattice (E8P)** | **8** | **最优** | **快** | **2-4 bit** |
| K-means (unstructured) | 任意 | 理论最优 | **慢** | 任意 |

E₈ 在同比特率下的 elementwise MSE 显著低于所有其他结构化码本。

## 与其他量化方法的对比

- **[[GPTQ]] / RTN**：标量量化，简单但效率低
- **[[AWQ]]**：标量量化 + 等价缩放，仍是 SQ
- **[[AQLM]]**：非结构化 VQ（learned codebook），以加法组合多 codebook 向量编码，精度最高但推理 kernel 支持有限
- **QuIP#**：结构化 VQ（E₈），码本小，推理快

AQLM 与 QuIP# 代表了两条不同的 VQ 路线：AQLM 用 learned codebook 自由拟合权重分布（更高精度但更慢），QuIP# 用固定数学结构的格码本（更快推理但精度略低）。在 2-bit 下 AQLM 全面超越 QuIP#（LLAMA2-70B: 3.94 vs 4.16 PPL）。

## 影响与展望

E₈ VQ 证明了**高维结构化向量量化**在 LLM 压缩中的巨大潜力。其成功的关键前提是 [[Rotation-based Quantization|旋转 incoherence processing]] 将权重变为球形分布——这两项技术构成了一个完整的理论链条：incoherence → 球形分布 → 格码本。
