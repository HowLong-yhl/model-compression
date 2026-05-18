---
title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
aliases: [GQA, Grouped-Query Attention, Ainslie 2023]
source_type: paper
source_url: "https://arxiv.org/abs/2305.13245"
source_date: 2023-05-22
ingested: 2026-04-28
venue: EMNLP 2023
authors: [Joshua Ainslie, James Lee-Thorp, Michal de Jong, Yinfei Yang, Siddhartha Rao Kamalakara, Manzil Zaheer]
affiliation: Google Research
code: null
tags: [model-compression, kv-cache, attention-mechanism, grouped-query-attention, multi-query-attention, architecture-design, inference-efficiency]
---

## 核心要点

Grouped-Query Attention (GQA) 是 Google Research 提出的一种介于 **Multi-Head Attention (MHA)** 和 **Multi-Query Attention (MQA)** 之间的注意力机制，旨在平衡推理速度与模型质量。MQA 通过让所有 query head 共享一组 key-value head 来大幅减少 KV cache 和内存带宽需求，但可能导致质量下降和训练不稳定。GQA 提出将 query heads 分成 **$G$ 个组**，每组共享一对 key-value head，实现了质量-速度的灵活折中。

更重要的是，作者提出了一种 **uptraining** 方法：将已有的 MHA 模型 checkpoint 通过仅 **5% 的原始预训练计算量** 转换为 GQA 模型，避免了从头训练的巨大成本。实验表明，uptrained GQA 在质量上 **接近 MHA**，在速度上 **接近 MQA**，是一个极具实用价值的工业级方案。GQA 已被 LLaMA-2 70B、LLaMA-3、Mistral 等主流模型广泛采用。

## 方法详解

### 从 MHA 到 MQA 到 GQA 的演进

**Multi-Head Attention (MHA)**：
- 每个 attention head 有独立的 $Q$, $K$, $V$ 投影
- $H$ 个 query heads 对应 $H$ 组 KV heads
- KV cache 大小：$H \times d_k \times L$（$L$ 为序列长度）

**Multi-Query Attention (MQA)**（Shazeer, 2019）：
- 所有 query heads 共享 **1 组** KV heads
- KV cache 缩减为 $1 \times d_k \times L$（减少 $H$ 倍）
- 问题：质量下降，训练不稳定

**Grouped-Query Attention (GQA)**：
- 将 $H$ 个 query heads 分为 $G$ 组
- 每组共享 1 对 KV heads
- KV cache 大小：$G \times d_k \times L$
- 当 $G = H$ 时退化为 MHA；$G = 1$ 时退化为 MQA

```
MHA:  H query heads -> H KV heads    (G = H)
GQA:  H query heads -> G KV heads    (1 < G < H)
MQA:  H query heads -> 1 KV head     (G = 1)
```

### Uptraining 方法

从现有 MHA checkpoint 转换为 GQA 的流程：

**Step 1: KV Head 合并**
将 MHA 的 $H$ 个 KV head 合并为 $G$ 组：
- **Mean pooling（推荐）**：每组内的 KV head 参数取平均
  $$K_g = \frac{1}{|S_g|} \sum_{h \in S_g} K_h$$
- **选择第一个 head**：直接取每组的第一个 KV head 参数

实验发现 **mean pooling** 效果更好，因为它保留了组内所有 head 的信息。

**Step 2: Uptraining**
- 使用合并后的参数初始化 GQA 模型
- 在原始预训练数据上继续训练 **$\alpha$ 比例** 的 tokens
- 推荐 $\alpha = 0.05$（即仅需 5% 的原始预训练计算量）

### 为什么 GQA 有效？

1. **KV cache 减少与质量的非线性关系**：从 MHA 到 MQA 的质量下降并非线性的。即使 $G$ 远小于 $H$（如 $G = 8$, $H = 64$），质量也只有微小损失。
2. **内存带宽瓶颈**：推理时的主要瓶颈是从 HBM 加载 KV cache 到计算单元的内存带宽。将 KV heads 从 $H$ 减少到 $G$ 直接降低了 bandwidth 需求。
3. **Uptraining 的高效性**：MHA 中同一组的 KV heads 本身就存在高度冗余（相关性高），mean pooling 很好地保留了核心信息。

### GQA 的典型配置

| 模型 | Total Heads $H$ | KV Groups $G$ | KV Cache 减少 |
|------|-----------------|---------------|--------------|
| LLaMA-2-70B | 64 | 8 | 8x |
| LLaMA-3-8B | 32 | 8 | 4x |
| LLaMA-3-70B | 64 | 8 | 8x |
| Mistral-7B | 32 | 8 | 4x |
| Gemma-7B | 16 | 1 (MQA) | 16x |

## 实验结果

### 质量对比（T5 XXL, 翻译/摘要任务）

| 方法 | KV Heads | 质量 (相对 MHA) | 推理速度 (相对 MHA) |
|------|----------|----------------|-------------------|
| MHA | $H$ | 基线 | 1x |
| MQA (from scratch) | 1 | -1.5% | ~1.7x |
| MQA (uptrained) | 1 | -0.8% | ~1.7x |
| **GQA (uptrained, G=8)** | 8 | **-0.2%** | **~1.5x** |

### Uptraining 计算量分析

| Uptraining 比例 | GQA 质量损失 | 计算成本 |
|----------------|------------|---------|
| 0% (直接转换) | -2.1% | 0 |
| 5% | -0.2% | 低 |
| 10% | -0.1% | 中 |
| 100% (从头训练) | ~0% | 极高 |

**关键发现**：仅 5% 的 uptraining 即可恢复大部分质量损失。

### KV Head 合并策略对比

| 合并策略 | 质量 (相对 MHA) |
|---------|----------------|
| 随机选择 1 个 head | -1.2% |
| 选择第 1 个 head | -0.9% |
| **Mean pooling** | **-0.2%** |

### 推理速度提升

在 A100 GPU 上的实测结果：
- Batch size = 1：GQA-8 vs MHA 加速 **1.5x**
- Batch size = 8：GQA-8 vs MHA 加速 **1.3x**
- Batch size = 32：差距缩小（compute-bound 而非 memory-bound）

加速效果在 **小 batch size** 和 **长序列** 时更显著，因为此时推理是内存带宽瓶颈。

## 局限性

1. **需要 uptraining**：虽然只需 5% 的计算量，但对于 >100B 的大模型，这仍然是一笔不小的开销。对于没有原始预训练数据的场景，uptraining 可能受限。
2. **不改变 prefill 速度**：GQA 主要加速 decoding 阶段（memory-bound），对 prefill 阶段（compute-bound）的加速有限。
3. **组数 $G$ 的选择是硬件相关的**：最优的 $G$ 值取决于硬件的内存带宽与计算吞吐的比率，需要针对目标部署平台进行 profiling。
4. **质量损失在某些任务上不可忽略**：虽然平均质量损失很小，但在某些对 KV 表达能力敏感的任务（如需要精细 retrieval 的长文本理解）上可能更显著。
5. **架构修改不可逆**：一旦将 MHA 转换为 GQA 并部署，无法在推理时灵活切换回 MHA 以获得更高质量。

## 与已有知识的关联

- **与 KV Cache Eviction 方法的关系**：GQA 从 **架构层面** 减少 KV cache（减少 KV head 数量），而 [[StreamingLLM]]、[[Scissorhands]]、[[H2O]]、[[SnapKV]]、[[PyramidKV]] 从 **推理层面** 减少 KV cache（减少保留的 token 数量）。两类方法 **完全正交**，可以叠加使用。例如 LLaMA-3-8B 使用 GQA (4x KV 减少) + SnapKV (5x token 减少) 可实现 20x 总压缩。
- **与 [[KV Cache Management]] 的关系**：GQA 是 KV cache 管理的 **源头方法** -- 它直接减少了需要管理的 KV cache 总量，使得后续的 eviction/quantization 方法在更小的 cache 上操作。
- **与 [[DeepSeek-V2-MLA]] 的关系**：MLA 是 GQA 思路的进一步延伸 -- 不只是简单地将 KV heads 分组共享，而是通过 **低秩投影** 将所有 KV heads 压缩到一个低维潜在空间。MLA 可以被视为"连续版"的 GQA，允许更激进的压缩。
- **与 KV Cache Quantization 的关系**：GQA 与量化方法（[[KIVI]]、[[KVQuant]]、[[TurboQuant]]）正交，可以组合使用。GQA 减少 head 数量 * 量化减少每个 head 的 bit 数。
- **与 [[GPTQ]]、[[AWQ]] 的关系**：这些是权重量化方法，与 GQA 的 KV cache 压缩目标不同但互补。部署时可以同时使用 GQA（KV cache 减少）+ AWQ（权重量化）+ KIVI（KV cache 量化）三层压缩。

## 引用关系

### 被引用 (本文启发的后续工作) ->
- -> [[DeepSeek-V2-MLA]]：将 GQA 的 group 共享思想推广为低秩投影
- -> LLaMA-2/3、Mistral 等主流模型直接采用 GQA 架构
- -> [[StreamingLLM]]、[[SnapKV]]、[[PyramidKV]] 等 KV cache eviction 方法在 GQA 模型上进行评测
- -> [[KIVI]]、[[KVQuant]] 等 KV cache 量化方法在 GQA 基础上进一步压缩

### 引用的前置工作 <-
- <- Multi-Query Attention (Shazeer, 2019)：GQA 的直接前驱
- <- Multi-Head Attention (Vaswani et al., 2017)：原始 Transformer 注意力机制
- <- T5 (Raffel et al., 2020)：实验基础模型
- <- Fast Transformer Decoding (Shazeer, 2019)：MQA 的提出
