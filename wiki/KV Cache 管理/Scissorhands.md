---
title: "Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression"
aliases: [Scissorhands, Liu 2024 Scissorhands]
source_type: paper
source_url: "https://arxiv.org/abs/2305.17118"
source_date: 2023-05-26
ingested: 2026-04-28
venue: ICLR 2024
authors: [Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, Anshumali Shrivastava]
affiliation: Rice University
code: "https://github.com/Zefan-Cai/Scissorhands"
tags: [model-compression, kv-cache, token-eviction, attention-sparsity, inference-efficiency, pivotal-token]
---

## 核心要点

Scissorhands 提出了 **Persistence of Importance Hypothesis（重要性持续性假设）**：在 LLM 的自回归生成过程中，某些 **pivotal tokens**（关键 token）在历史步骤中对注意力产生了显著影响，这些 token 在未来的生成步骤中 **也将持续保持其重要性**。换言之，token 的重要性具有时间持续性 -- 过去重要的 token 未来仍然重要。

基于这一假设，Scissorhands 设计了一种无需重训练的推理时 KV cache 压缩方案：在每一步生成时，根据历史注意力分数识别 pivotal tokens，优先保留这些 token 的 KV 对，驱逐非关键 token，从而在严格的内存预算下维持模型质量。实验表明，Scissorhands 可以在 **不损害模型质量** 的前提下将 KV cache 压缩至原始大小的 **20%**（即 5x 压缩），结合 4-bit 量化可实现高达 **20x** 的总压缩比。

## 方法详解

### Persistence of Importance Hypothesis

作者首先通过实验验证了这一核心假设：

1. **Pivotal token 定义**：在生成步骤 $t$，如果某个 token $i$ 的注意力分数 $\alpha_{t,i}$ 超过某个阈值（例如位于 top-k），则 token $i$ 被认为是步骤 $t$ 的 pivotal token。
2. **实证验证**：作者在多种 LLM（OPT、LLaMA、Falcon）上统计发现，如果某个 token 在步骤 $t$ 被标记为 pivotal，那么它在步骤 $t+1, t+2, \ldots$ 继续保持 pivotal 状态的概率非常高（通常 > 90%）。
3. **对比实验**：随机选择 token 保留 vs. 基于 importance 选择，后者在相同压缩率下的 perplexity 远优于前者。

### Scissorhands 算法

Scissorhands 的工作流程如下：

**Phase 1: 正常运行阶段（Cache 未满）**
- 正常执行 attention 计算，将每步生成的 KV 对加入 cache
- 同时记录每个 token 在各 attention head 中的历史注意力分数

**Phase 2: 压缩阶段（Cache 超预算）**
当 KV cache 大小超过预设预算 $B$ 时，执行以下操作：

```
1. 对于每个 attention head：
   a. 计算每个 token 的 importance score = 历史窗口内的注意力分数聚合
   b. 选择 top-B 个 token 作为保留集合
   c. 驱逐（evict）其余 token 的 KV 对

2. 特殊保留规则：
   a. 始终保留最近的 r 个 token（local window）
   b. 始终保留 attention sink tokens（前几个 token）
```

### 重要性评分机制

Scissorhands 使用 **历史注意力分数** 来量化 token 重要性：

- **单步 importance**：token $i$ 在步骤 $t$ 的 importance 为其在当前 query 下的 attention weight $\alpha_{t,i}$
- **累积 importance**：基于最近 $w$ 步的注意力分数进行聚合（取平均或最大值）
- **Per-head 独立决策**：不同的 attention head 可以独立选择保留不同的 token 子集，因为不同 head 的注意力模式差异显著

### 与量化的正交结合

Scissorhands 的 token eviction 策略与 KV cache 量化方法正交，可以叠加使用：
- Eviction 将 token 数量从 $N$ 减少到 $B$（5x）
- 4-bit 量化将每个 token 的存储从 FP16 减少到 INT4（4x）
- 组合效果：$5 \times 4 = 20$x 总压缩

## 实验结果

### KV Cache 压缩效果

| 模型 | 压缩率 | Perplexity (原始) | Perplexity (Scissorhands) | 质量损失 |
|------|--------|-------------------|---------------------------|---------|
| OPT-6.7B | 5x | ~10.8 | ~11.0 | < 2% |
| OPT-13B | 5x | ~10.1 | ~10.3 | < 2% |
| LLaMA-7B | 5x | ~7.5 | ~7.7 | < 3% |
| Falcon-7B | 5x | ~8.0 | ~8.2 | < 3% |

### Pivotal Token 持续性验证

- 在 OPT-6.7B 上，一个 token 如果在步骤 $t$ 是 pivotal（top-5%），其在下一步继续保持 pivotal 的条件概率约为 **93-97%**。
- 这一特性在不同层和不同 attention head 之间普遍存在。

### 下游任务表现

- 在长文本生成、对话、摘要等任务上，20% cache budget 下的 Scissorhands 输出质量与全量 cache 相当。
- 在某些任务上，适度的 cache 压缩甚至略微提升了输出质量（类似于正则化效果）。

### 结合量化的总效果

| 方法 | 总压缩率 | 质量保持 |
|------|---------|---------|
| Scissorhands only | 5x | 接近无损 |
| 4-bit quant only | 4x | 接近无损 |
| **Scissorhands + 4-bit** | **20x** | 轻微下降 |

## 局限性

1. **需要 attention score 作为 importance 信号**：每步生成时需要额外记录注意力分数用于后续决策，增加了一定的计算和存储开销。
2. **阈值/预算参数敏感性**：cache 预算 $B$ 和 local window 大小 $r$ 的选择需要根据任务和模型进行调优。
3. **假设可能在极端场景下失效**：对于 attention 模式快速变化的任务（如需要频繁切换关注目标的多跳推理），pivotal token 的持续性假设可能不完全成立。
4. **Per-head 独立决策的开销**：每个 attention head 独立维护保留集合，在 head 数量大的模型上管理复杂度较高。
5. **仅在推理阶段压缩**：不改变模型结构，无法在训练阶段获得加速。

## 与已有知识的关联

- **与 [[StreamingLLM]] 的关系**：StreamingLLM 发现了 attention sink 并固定保留初始 token，Scissorhands 则更进一步，通过 importance score 动态识别所有值得保留的 token（包括但不限于 sink tokens）。两者都是 eviction-based KV cache 压缩方法，但 Scissorhands 的选择策略更灵活。
- **与 [[H2O]] 的关系**：H2O (Heavy Hitter Oracle) 与 Scissorhands 高度相关 -- 两者都基于注意力分数来选择保留的 token。区别在于 H2O 使用累积注意力分数（Heavy Hitter 概念），而 Scissorhands 更强调 importance 的 **时间持续性**。两者的核心 insight 互相印证。
- **与 [[SnapKV]] 的关系**：SnapKV 在 prefill 阶段一次性选择重要 token，而 Scissorhands 在每步生成时动态调整。SnapKV 可以被视为 Scissorhands 思想的简化版本。
- **与 [[Token Eviction]] 的关系**：Scissorhands 是 token eviction 策略的代表性工作，其 "persistence of importance" 假设成为后续 eviction 方法的重要理论基础。
- **与 [[Attention Sparsity]] 的关系**：Scissorhands 利用了注意力的稀疏性 -- 只有少量 pivotal tokens 真正对生成质量有影响，其余 token 可以安全驱逐。
- **与 [[KV Cache Management]] 的关系**：与 quantization-based 方法（[[KIVI]]、[[KVQuant]]）正交，可以组合使用实现更高压缩比。

## 引用关系

### 被引用 (本文启发的后续工作) ->
- -> [[SnapKV]]：简化了 token selection 策略，在 prefill 阶段一次性完成
- -> [[PyramidKV]]：引入层级自适应的 cache 预算分配
- -> [[H2O]]：类似的 attention-based eviction 策略
- -> [[ShadowKV]]：结合 offloading 和 eviction 的混合策略

### 引用的前置工作 <-
- <- [[StreamingLLM]]：attention sink 现象（同期工作，互相引用）
- <- Multi-Query Attention / [[GQA]]：从架构层面减少 KV cache 的先驱
- <- Attention Sparsity 研究（Correia et al., 2019; Sukhbaatar et al., 2019）
