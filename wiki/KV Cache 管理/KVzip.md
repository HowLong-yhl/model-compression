---
title: "KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction"
aliases: [KVzip, Kim 2025]
source_type: paper
source_url: "https://arxiv.org/abs/2505.08297"
source_date: 2025-05-13
ingested: 2026-04-28
venue: NeurIPS 2025
authors: [Jang-Hyun Kim, Jinuk Kim, Sangwoo Kwon, Jae W. Lee, Sangdoo Yun, Hyun Oh Song]
affiliation: Seoul National University, NAVER AI Lab
code: "https://github.com/snu-mllab/KVzip"
tags: [model-compression, method, kv-cache, eviction, query-agnostic, inference]
---

## 核心要点

KVzip 指出现有 eviction 方法（[[H2O]]、[[SnapKV]]、[[PyramidKV]]）都是 **query-aware** 的——它们基于当前 query 的 attention 信息判断 KV 重要性。这导致两个问题：(a) 多 query 场景下压缩后的 cache 对首个 query 过拟合、对后续 query 退化；(b) 每个新 query 都需要重新 prefill。

KVzip 提出 **query-agnostic** 的 KV 重要性评分方法——基于 **context reconstruction**（让 LLM 复述原始上下文），能够评估每个 KV 对对"保留整体上下文信息"的贡献。压缩后的 cache 可跨不同 query 复用。

## 方法详解

### 核心思路：Context Reconstruction

**直觉**：Transformer LLM 天然是 encoder-decoder——将上下文编码为 KV cache。如果某些 KV 对对"重建原始上下文"至关重要，那么它们对任意未来 query 也应该是重要的（类似 Zip 压缩和自监督学习的重建目标）。

### 重要性评分

1. 在原始上下文后拼接一个 "repeat prompt"（如 "Repeat the previous context:"）
2. 用 teacher-forced decoding 跑一次前向传播——让 LLM 基于 cached KV 重建原始上下文
3. 每个 KV 对的重要性分数 = 它在重建过程中获得的**最大 attention score**（跨所有 grouped query 和 input position 取 max）

### Chunked Scoring（可扩展性）

直接计算是 $O(n_c^2)$。KVzip 将上下文分成固定大小的 chunk（m=2K token），逐 chunk 独立重建并计算重要性分数。复杂度降为 $O(m \cdot n_c)$，即线性于上下文长度。峰值内存 $O(m^2)$，与上下文长度无关。

### Eviction 策略

- 全局（跨所有 attention head）选择 top-r% 的 KV 对保留
- 允许 **non-uniform head budget**——不同 head 可能保留不同数量的 KV 对
- System prompt 的 KV 完整保留

### Context-Independent 模式

可进一步简化为**静态 head-level 重要性评分**——用单个 calibration 文本计算每个 head 的重要性，部署后零压缩开销。相比 DuoAttention 需要数小时 GPU 优化，KVzip 仅需几分钟前向传播。

## 实验结果

### 多 Query 复用（SQuAD, LLaMA3.1-8B）

| 方法 | 30% Cache | 60% Cache | 90% Cache |
|------|----------|----------|----------|
| Full KV | 100% | 100% | 100% |
| H2O | ~40% | ~65% | ~80% |
| SnapKV-reuse | ~30% | ~40% | ~70% |
| **KVzip** | **~95%** | **~98%** | **~100%** |

在多 query 场景下，query-aware 方法严重退化（SnapKV-reuse 在 60% cache 下仅 40%），KVzip 在 30% cache 下仍保持 95%。

### 12 Benchmark 数据集（Qwen2.5-7B-1M）

KVzip 在 20-30% cache ratio 下跨检索（NIAH, Code.RepoQA）、理解（SQuAD, GSM8K）和冗余（Summary, ICL）三大类任务均保持近无损性能。基线方法在 90% retention 下即开始退化。

### 与量化的兼容性

16-bit KV (16.3GB) → 4-bit 量化 + 70% eviction = **1.2GB**，精度几乎无退化。证明量化和 eviction 的正交互补性。

### 推理效率

| 指标 | 效果 |
|------|------|
| FlashAttention 解码延迟 | **~2× 降低** |
| KV cache 大小 (124K context) | 16.3GB → **~5GB** (30% ratio) |

## 局限性

- **Reconstruction 前向传播开销**：需要额外一次完整的 teacher-forced forward pass 来计算重要性分数
- **隐私泄露风险**：论文发现压缩后的 cache 可能更容易泄露上下文中的隐私信息（如电话号码），因为重建过程倾向于保留"信息密度高"的 token
- **Chunk 边界效应**：chunked scoring 可能在 chunk 边界处遗漏跨 chunk 的重要依赖
- **Repeat prompt 的设计**：重建效果依赖于 repeat prompt 的措辞，可能对不同模型需要微调

## 与已有知识的关联

### Query-Aware vs Query-Agnostic 的范式转换

KVzip 揭示了 KV cache eviction 中一个根本性的设计维度——query 依赖性。[[H2O]] 和 [[SnapKV]] 都是 query-aware：H2O 基于生成时的 attention 动态 evict，SnapKV 用 observation window 的 attention 做 prefill 压缩。这在单 query 场景有效，但在多 query 场景（同一文档回答多个问题、API serving 中的 prompt caching）下严重退化。

KVzip 的 context reconstruction 提供了一种 query-agnostic 的替代范式，[[TriAttention]] 从另一个角度（pre-RoPE Q/K 中心的离线统计）也实现了 query-agnostic。

### 与 Attention Sparsity 的关系

KVzip 利用了一个更深层的 sparsity 发现：reconstruction attention（让 LLM 复述上下文时的 attention）比 prefill attention（正常前向传播）更加稀疏——这使得 reconstruction-based scoring 更能区分 KV 对的重要性。

## 引用关系

← 改进自：[[H2O]]（从 query-aware 到 query-agnostic），[[SnapKV]]（解决 multi-query 退化）
← 理论基础：Context reconstruction 思想（类比 Zip 压缩、自监督学习的 reconstruction objective）
→ 贡献：首个 query-agnostic KV cache eviction 方法
→ 概念关联：[[Token Eviction]]（eviction 方法的 query-agnostic 代表），[[KV Cache Management]]
→ 与量化兼容：4-bit KV + 70% eviction 可叠加使用
