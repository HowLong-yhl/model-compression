---
title: "SnapKV: LLM Knows What You are Looking for Before Generation"
aliases: [SnapKV, Li 2024 SnapKV]
source_type: paper
source_url: "https://arxiv.org/abs/2404.14469"
source_date: 2024-04-22
ingested: 2026-04-28
venue: NeurIPS 2024
authors: [Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Patrick Lewis, Hanchen Ye, Tianle Cai, Deming Chen]
affiliation: UIUC, Cohere, Princeton University
code: "https://github.com/FasterDecoding/SnapKV"
tags: [model-compression, method, kv-cache, eviction, prefill, inference]
---

## 核心要点

SnapKV 发现 LLM 在生成之前就已经"知道"哪些输入 token 是重要的——prompt 最后一个窗口（observation window）的 attention 模式能稳定预测生成阶段的 KV 重要性。据此提出在 **prefill 阶段**一次性压缩 prompt 的 KV cache，用 observation-window voting + pooling-based clustering 选择重要 token。

与 [[H2O]] 的关键区别：H2O 在 generation 阶段逐步 eviction，SnapKV 在 prefill 完成后一次性压缩整个 prompt cache。这对长 prompt + 短生成的实际场景（聊天、文档问答）更有效。

## 两个核心观察

### 观察一：生成前即可识别重要模式

使用 prompt 最后 L_obs 个 token（observation window）作为 query，计算它们对所有 prefix token 的 attention 权重。这些权重模式与实际生成阶段使用的 attention 模式高度一致——重叠率在多种模型上超过 90%。

### 观察二：重要模式在生成过程中保持稳定

observation window 选出的重要 token 在整个生成过程中持续保持其重要性——不会出现"前几步重要、后面不重要"的不一致。

## 方法详解

### 算法流程（fine-tuning-free）

**Step 1：Voting**

对 observation window 中的 L_obs 个 query，计算每个 prefix token 在每个 attention head 的累计 attention 权重：

$$\text{score}(i, h) = \sum_{q \in \text{obs\_window}} A_{q,i}^{(h)}$$

这产生每个 head 上 prefix token 的重要性排名。

**Step 2：Pooling-based Clustering**

直接选 top-k 单个 token 会破坏上下文完整性（例如一段关键信息可能跨越多个连续 token）。SnapKV 用 **1D max/average pooling**（kernel size 5-7）对重要性分数做平滑，使选择倾向于保留连续的 token 簇而非孤立 token。

**Step 3：Compression**

每个 head 独立选择 top-k prefix token（k = floor(p × L_prefix)），gather 对应的 K/V 向量，与 observation window 的 KV 拼接。压缩后的 cache 用于后续所有生成步骤。

### 默认配置

- Observation window size: L_obs = 32（prompt 最后 32 个 token）
- Pooling kernel size: 5 或 7
- Compression ratio p: 可调，典型值使 cache 降至 1024-4096 token

## 实验结果

### 长上下文能力（Needle-in-a-Haystack）

在 LWM-Text-Chat-1M 上，SnapKV 以 1024 token cache size 在单张 A100-80GB 上处理最长 **380K token** 输入（380× 压缩），在 140K 以内保持正确检索。

### LongBench 精度（16 个数据集）

| 模型 | 方法 | Cache Size | Avg. Score |
|------|------|-----------|------------|
| Mistral-7B | Full KV | ∞ | baseline |
| | SnapKV | 4096 | ≈ Full KV |
| | H2O | 4096 | 显著退化 |
| Mixtral-8x7B | Full KV | ∞ | baseline |
| | SnapKV | 4096 | ≈ Full KV |
| | H2O | 4096 | Single-Doc QA 48.02 vs 52.62，Multi-Doc QA 34.76 vs 47.71 |

SnapKV 在 4096 cache 下几乎无损，H2O 则在文档问答类任务上明显退化。

### 系统性能

| 指标 | 效果 |
|------|------|
| 解码加速 (16K input, BS=2) | **3.6×** |
| 内存效率提升 | **8.2×**（从 16K OOM 扩展到 131K） |
| 压缩率 (1024 cache) | **92%** 压缩，精度损失可忽略 |

### 消融实验

**Pooling 的关键性**：去掉 pooling 后，Needle-in-a-Haystack 在 10K+ 输入长度即失败——因为孤立选择 top-k token 会丢失关键信息片段的上下文。

## 局限性

- **Query-aware**：observation window 的 attention 模式依赖于 prompt 末尾内容，不同 query 可能选出不同 token。在多 query 复用场景下（如同一文档回答多个问题），压缩后的 cache 对第一个 query 优化但对后续 query 次优——[[KVzip]] 专门解决了这个问题
- **一次性压缩，不可恢复**：被驱逐的 prefix token KV 永久丢失
- **Observation window 大小的折中**：L_obs 太小则投票不充分，太大则计算开销增加。默认 32 是经验值
- **未考虑层间差异**：所有层使用相同的 compression ratio——[[PyramidKV]] 改进为层自适应预算

## 与已有知识的关联

### Prefill-time vs Generation-time Eviction

SnapKV 和 [[H2O]] 代表了 eviction 时机的两个端点：
- H2O 在 generation 阶段逐步 eviction（每个解码步检查并驱逐）
- SnapKV 在 prefill 完成后一次性压缩

在实际部署中（长 prompt + 短回复），prefill 阶段的 KV cache 占主导，SnapKV 的策略更高效。但 SnapKV 无法在 generation 阶段进一步压缩。

### 与 KV Cache 量化的正交性

SnapKV 减少 token 数量，量化（[[KIVI]]、[[KVQuant]]）减少每个 token 的比特数。两者可组合：先 SnapKV 选出重要 token，再对保留的 token 做低比特量化。

## 引用关系

← 改进自：[[H2O]]（从 generation-time 扩展到 prefill-time 压缩）
← 相关发现：[[Attention Sparsity]]（SnapKV 利用 attention 的集中性来识别重要 token）
→ 贡献：首个系统化的 prefill-time KV cache 压缩方案
→ 被改进：[[KVzip]]（解决 query-aware 在多 query 场景的局限），[[TriAttention]]（用 pre-RoPE 理论替代经验性 observation window），[[PyramidKV]]（引入层自适应预算）
→ 概念关联：[[Token Eviction]]，[[KV Cache Management]]
