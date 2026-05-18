---
title: Attention Sink
aliases: [注意力汇聚, Sink Token, Initial Token Attention]
tags: [model-compression, concept, attention, kv-cache]
created: 2026-04-28
updated: 2026-04-28
sources: [StreamingLLM, H2O]
---

## 定义

Attention Sink 是指 Transformer LLM 中一种普遍现象——**序列起始 token（尤其是第一个 token / BOS token）持续获得不成比例的高 attention score**，无论其语义内容是什么。这由 [[StreamingLLM]]（MIT Han Lab, ICLR 2024）首次系统发现和命名。

## 核心特征

### 现象描述

- 即使第一个 token 是无意义的 padding 或标点符号，它也获得高 attention
- 这种模式跨层、跨 head 一致出现
- 移除 sink token 会导致 softmax 分布崩塌和模型输出严重退化

### 成因分析

Attention sink 并非因为这些 token 语义重要，而是 softmax 的数学特性导致的**隐式偏置**：

softmax 要求所有 attention 权重之和为 1。当模型"不需要"关注任何特定 token 时（例如某些 head 在某些位置不需要检索信息），它需要一个"垃圾桶"来承接多余的 attention 概率质量。初始 token 因为其固定位置和训练动态，被模型学习为这个默认的注意力接收器。

### 与 Heavy Hitter 的关系

[[H2O]] 的 Heavy Hitter 概念与 Attention Sink 高度相关但不完全等同：
- **Attention Sink**（StreamingLLM）：强调初始 token 的特殊地位——它们总是 sink，即使无语义价值
- **Heavy Hitter**（H2O）：泛指累计 attention score 最高的 token 集合——可能包括 sink token 和语义上确实重要的 token

实际上，在多数情况下初始 token 是 Heavy Hitter 集合中的成员。

## 对 KV Cache 管理的启示

Attention Sink 的发现有直接的工程启示——**任何 KV cache eviction 策略都必须保留 sink token**，否则模型输出会崩溃。

| 方法 | 如何处理 sink token |
|------|-------------------|
| [[StreamingLLM]] | 显式保留 sink token + sliding window |
| [[H2O]] | 自动保留（sink token 累计 attention 分数最高，自然被选为 heavy hitter） |
| [[SnapKV]] | 自动保留（observation window voting 自然给 sink token 高分） |
| [[KVzip]] | 显式保留 system prompt 的 KV |
| [[PyramidKV]] | 兼容 sink 保留策略 |

### 工程应用

StreamingLLM 提出的 "sink + window" 策略已成为长上下文推理的基础配置：
- 保留前 k 个 sink token（通常 k=4）
- 保留最近 n 个 token 的 sliding window
- 对位置编码做 re-mapping（使 window 内 token 的位置 ID 连续）

这使 LLM 可以处理**无限长度**的流式输入，代价是丢失中间上下文。

## 与 WKVQuant 的关联

WKVQuant（ACL 2025）发现量化场景下 outlier token（包括 BOS/sink token 和 attention 汇聚 token）对 KV cache 量化误差影响巨大。保留这些 token 的全精度 KV 可以显著提升量化精度——这是 Attention Sink 概念在量化领域的直接应用。

## 引用关系

← 发现者：[[StreamingLLM]]（首次命名和系统化）
← 相关现象：[[Attention Sparsity]]（sink 是 sparsity 的一个极端表现），[[Emergent Outlier Features]]（可能的底层联系）
→ 应用：[[Token Eviction]]（所有 eviction 方法需保留 sink），[[KV Cache Management]]
→ 工程意义：streaming inference、长上下文部署的基础策略
