---
title: "Efficient Streaming Language Models with Attention Sinks"
aliases: [StreamingLLM, Xiao 2024]
source_type: paper
source_url: "https://arxiv.org/abs/2309.17453"
source_date: 2023-09-29
ingested: 2026-04-28
venue: ICLR 2024
authors: [Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis]
affiliation: MIT, Meta AI (FAIR), CMU, NVIDIA
code: "https://github.com/mit-han-lab/streaming-llm"
tags: [model-compression, kv-cache, attention-sink, streaming-inference, token-eviction, long-context, inference-efficiency]
---

## 核心要点

StreamingLLM 是首个系统性揭示 **Attention Sink** 现象并据此设计无限长度流式推理方案的工作。作者发现，在 autoregressive LLM 中，即使初始 token 在语义上与当前生成无关，它们也会持续吸引大量注意力权重 -- 这被称为 "attention sink"。这一现象的根本原因在于 Softmax 运算的性质：模型需要将注意力分数归一化为概率分布，当没有明确的语义关注目标时，模型倾向于将多余的注意力"倾倒"到初始位置的 token 上。

基于这一发现，StreamingLLM 提出了一种极简而高效的方案：在 KV cache 中仅保留少量 **sink tokens**（通常为前 4 个 token）加上一个固定大小的 **sliding window**（滑动窗口），即可让有限窗口的 LLM 稳定处理 **无限长度** 的文本流，无需任何微调。这一方法在 Llama-2、MPT、Falcon、Pythia 等多种模型上均实现了稳定的流式部署，相比滑动窗口基线速度提升最高 22.2 倍。

## 方法详解

### Attention Sink 现象的发现

作者通过分析多种预训练 LLM（Llama-2、MPT、Falcon、Pythia）的注意力分布，发现了一个普遍性的现象：

1. **初始 token 吸引异常高的注意力**：无论输入序列有多长，第一个（或前几个）token 始终获得不成比例的高注意力分数，即使这些 token 是无语义的标点或特殊符号。
2. **去除 sink token 导致模型崩溃**：当 sliding window eviction 策略丢弃了初始 token 后，模型的困惑度（perplexity）会急剧上升甚至发散，因为注意力的归一化分布被破坏。
3. **Sink 现象与语义无关**：将初始 token 替换为任意文本，sink 效应依然存在，说明这是由位置（position）而非内容（content）驱动的。

### StreamingLLM 框架

StreamingLLM 的核心设计非常简洁：

```
KV Cache = [Sink Tokens (前4个)] + [Sliding Window (最近L个)]
```

- **Sink Tokens**：保留序列最开始的 $k$ 个 token（默认 $k=4$）的 KV 对，作为注意力的"锚点"。
- **Sliding Window**：保留最近 $L$ 个 token 的 KV 对，提供局部上下文。
- **位置编码重映射**：关键创新点 -- 使用 **cache 内的相对位置** 而非原始文本中的绝对位置来计算位置编码。这解决了当中间 token 被驱逐后，位置 ID 不连续导致的 perplexity 爆炸问题。

### 训练阶段改进：带 Learnable Sink Token 的预训练

作者进一步提出，在预训练阶段引入一个专门的 **placeholder token** 作为 dedicated attention sink：
- 在每个训练样本的开头添加一个可学习的 sink token（如 `<sink>`）
- 模型学会将多余的注意力集中到这个专用 token 上
- 推理时仅需保留 1 个 sink token（而非 4 个），进一步减少 KV cache 开销

### 与 Window Attention 的关键区别

| 策略 | KV Cache 大小 | 长文本稳定性 | 需要重计算 |
|------|-------------|------------|----------|
| Dense Attention | $O(n)$ | 稳定 | 否 |
| Window Attention | $O(L)$ | 崩溃 | 否 |
| Sliding Window + Re-computation | $O(L)$ | 稳定 | 是（慢） |
| **StreamingLLM** | $O(k+L)$ | **稳定** | **否** |

## 实验结果

### 流式推理稳定性

| 模型 | 序列长度 | Window Attn PPL | StreamingLLM PPL |
|------|---------|----------------|-----------------|
| Llama-2-7B | 20K tokens | 发散 (>10^3) | ~8.0 (稳定) |
| Llama-2-13B | 20K tokens | 发散 | ~7.2 (稳定) |
| Llama-2-70B | 20K tokens | 发散 | ~5.8 (稳定) |
| Falcon-7B | 20K tokens | 发散 | 稳定 |
| MPT-7B | 20K tokens | 发散 | 稳定 |

### 速度提升

- 相比需要重新计算注意力的 Sliding Window + Re-computation baseline，StreamingLLM 在序列长度达 4M tokens 时实现了 **22.2x 加速**。
- Cache 大小仅需 ~2K tokens（4 sink + ~2000 recent），远小于完整 KV cache。

### Sink Token 数量分析

- 保留 **1 个** sink token 即可大幅改善稳定性
- 保留 **4 个** sink tokens 几乎在所有模型上达到最优
- 保留更多 sink tokens 边际收益递减

### Learnable Sink Token 效果

- 仅需 **1 个** learnable sink token 即可达到甚至超越保留 4 个初始 token 的效果
- 预训练阶段加入 sink token 对标准 benchmark 性能无影响

## 局限性

1. **无法利用被驱逐 token 的信息**：StreamingLLM 本质上是 eviction-based 方法，被丢弃的中间 token 信息完全丢失。对于需要长距离依赖的任务（如长文档问答），其性能受限于滑动窗口大小。
2. **不适用于需要全局上下文的任务**：在需要回顾整个文档（如摘要生成）的场景下，StreamingLLM 无法替代真正的长上下文模型。
3. **Sink 现象的理论理解不完整**：虽然作者给出了 Softmax 归一化的直觉解释，但 attention sink 的完整理论机制尚未被严格证明。
4. **与位置编码方案强耦合**：位置重映射策略的有效性依赖于 RoPE/ALiBi 等位置编码的具体实现，对于未来新的位置编码方案可能需要适配。

## 与已有知识的关联

- **与 [[H2O]] 的关系**：H2O 也利用了 attention sink 现象（"Heavy Hitter"即包含 sink tokens），但 H2O 使用累积注意力分数来动态选择保留哪些 token。StreamingLLM 更简洁，固定保留初始 token 而无需动态评分。两者的关键 insight 一致：**初始 token 对注意力稳定性至关重要**。
- **与 [[Scissorhands]] 的关系**：Scissorhands 同样在推理阶段进行 KV cache 压缩，但其基于 "pivotal token persistence" 假设来选择重要 token，而非专注于 sink tokens。StreamingLLM 的 sink 发现为 Scissorhands 的 token 重要性排序提供了补充视角。
- **与 [[SnapKV]] 的关系**：SnapKV 在 prefill 阶段基于观察窗口的注意力模式来选择保留的 KV 对，也受到了 attention sink 现象的启发。
- **[[Attention Sink]]** 概念由本文首次系统提出，后被 [[ShadowKV]]、[[SnapKV]]、[[H2O]] 等大量后续工作引用。
- **与 [[KV Cache Management]] 的关系**：StreamingLLM 是 eviction-based KV cache 管理的开创性工作之一，与 quantization-based（如 [[KIVI]]、[[KVQuant]]）和 offloading-based（如 [[ShadowKV]]）方法形成互补。
- **与 [[Emergent Outlier Features]] 的关系**：Attention sink 现象可能与模型中 emergent outlier 特征有关 -- 部分通道的激活值异常大，导致注意力集中到特定位置。

## 引用关系

### 被引用 (本文启发的后续工作) ->
- -> [[H2O]]：同样利用 attention sink，提出 Heavy Hitter Oracle
- -> [[SnapKV]]：基于注意力模式选择 KV cache，受 sink 现象启发
- -> [[ShadowKV]]：利用 sink token 特性设计 offloading 策略
- -> [[Scissorhands]]：互补的 token eviction 策略
- -> [[PyramidKV]]：层级自适应 cache 分配，考虑了不同层的 attention 分布差异

### 引用的前置工作 <-
- <- Multi-Query Attention / [[GQA]]：从架构层面减少 KV cache
- <- Sliding Window Attention (Beltagy et al., 2020)：StreamingLLM 的滑动窗口基线
- <- RoPE (Su et al., 2021)：StreamingLLM 重映射的位置编码基础
