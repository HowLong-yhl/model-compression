---
title: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models"
aliases: [H2O, Heavy-Hitter Oracle, Zhang 2023]
source_type: paper
source_url: "https://arxiv.org/abs/2306.14048"
source_date: 2023-06-24
ingested: 2026-04-28
venue: NeurIPS 2023
authors: [Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Re, Clark Barrett, Zhangyang Wang, Beidi Chen]
affiliation: UT Austin, Stanford, UC San Diego, UC Berkeley, Adobe, Meta AI (FAIR), CMU
code: "https://github.com/FMInference/H2O"
tags: [model-compression, method, kv-cache, eviction, attention-sparsity, inference]
---

## 核心要点

H2O 是 KV cache eviction 领域的**奠基性工作**，发现即使是密集训练的 LLM 在推理时注意力矩阵也高度稀疏（>95%），且少量 token 持续获得极高累计注意力分数（称为 **Heavy Hitter / H2**）。H2O 基于此提出一个简单高效的在线 eviction 策略——保留 Heavy Hitter + 最近 token，在仅 20% KV cache 预算下达到与全缓存接近的性能。

该方法被后续几乎所有 KV cache 管理工作（[[SnapKV]]、[[ShadowKV]]、[[KVzip]]、[[TriAttention]]、[[StreamingLLM]]）作为基准引用。

## 三个核心观察

### 观察一：推理时注意力天然稀疏

密集训练的 LLM 在推理时 attention 矩阵中 >95% 的权重接近零。这意味着每一步解码实际上只依赖约 5% 的 KV 对，其余可以安全移除。

### 观察二：Heavy Hitter 的幂律分布

将所有 token 按累计注意力分数排序，分布呈**幂律**——极少数 token 持续获得不成比例的高注意力（Heavy Hitter）。这些 H2 token 与文本中高频共现词强相关。移除 H2 token 会导致模型输出剧烈恶化。

### 观察三：局部贪心 ≈ 全局最优

仅基于已见 token 的注意力分数（local H2，不看未来 token）来判断重要性，与使用完整序列信息（global H2）的效果几乎一样好。这使得在线 eviction 可行。

## 方法详解

### H2O Eviction Policy

维护固定大小为 k 的 KV cache，由两部分组成：

```
KV Cache = Heavy Hitter 集合 (H2) ∪ 最近 token 窗口 (Recent)
```

**在线策略**：
1. 每步解码时，计算当前 token 对所有 cached token 的 attention score
2. 更新所有 token 的累计注意力分数
3. 当 cache 满时，驱逐累计分数最低的 token（不在 Recent 窗口内的）
4. 新 token 加入 cache

### 理论保证

H2O 将 KV cache 管理形式化为**动态子模极大化问题**（dynamic submodular maximization）：

$$f(S_i) \geq (1-\alpha)(1-1/e) \cdot \max_{|S| \leq k} f(S) - \beta$$

其中 f 是注意力分数的子模函数，α 和 β 分别是逼近误差项。贪心策略在理论上可达到近似最优。

## 实验结果

### 语言模型精度（20% KV cache 预算）

| 模型 | 方法 | COPA | OpenBookQA | PiQA | Avg. |
|------|------|------|------------|------|------|
| OPT-30B | Full KV | 85.00 | 44.80 | 80.52 | — |
| | Local (仅 recent) | 77.00 | 37.40 | 57.94 | — |
| | H2O | **85.00** | **43.80** | **79.22** | — |
| LLaMA-13B | Full KV | — | — | — | baseline |
| | H2O (20%) | — | — | — | ≈ Full KV |

H2O 在 20% 预算下几乎无损，而仅保留 recent token 的策略严重退化（PiQA 从 80.52 降至 57.94）。

### 系统性能

| 指标 | H2O vs 基线 |
|------|------------|
| vs DeepSpeed (OPT-30B) | **29× 吞吐量** |
| vs HuggingFace Accelerate | **29× 吞吐量** |
| vs FlexGen | **3× 吞吐量，1.9× 延迟降低** |

### 无限长度流式推理

H2O 支持 streaming 场景——在 4M token 序列上保持稳定的 perplexity，优于 [[StreamingLLM]]。

### 与量化的兼容性

H2O + 量化的组合几乎总是优于单独使用任一技术。eviction 减少 token 数量，量化降低每个 token 的比特数，两者正交互补。

## 局限性

- **Query-aware**：每步解码的 eviction 决策依赖当前 attention 模式，不同 query 可能需要不同的 H2 集合
- **信息不可恢复**：一旦 eviction，被驱逐的 KV 对永久丢失——在多轮对话中可能累积信息损失
- **未处理 prompt 压缩**：H2O 主要在 generation 阶段做 eviction，对长 prompt 的 prefill 压缩不如 [[SnapKV]] 有效
- **Head-uniform 预算**：所有 attention head 使用相同的 cache 大小，未利用不同 head 的重要性差异（后续 [[PyramidKV]] 改进了这一点）

## 与已有知识的关联

### 与 Attention Sparsity 的关系

H2O 的核心发现——推理时 attention >95% 稀疏——是 [[Attention Sparsity]] 概念的实证基础之一。这一稀疏性也被 [[KIVI]] 利用来论证 Value per-token 量化的可行性（稀疏 attention 权重限制了量化误差的传播）。

### 与 Attention Sink 的关系

H2O 发现的 Heavy Hitter 与 [[StreamingLLM]] 发现的 [[Attention Sink]] 高度相关——初始 token 通常是 Heavy Hitter。StreamingLLM 将此机制化为固定保留 sink token + sliding window；H2O 则用动态累计分数自适应选择。

## 引用关系

← 理论基础：[[Attention Sparsity]]（推理时 attention 天然稀疏），Submodular Optimization
→ 贡献：KV cache eviction 的奠基性工作，定义了 Heavy Hitter 概念
→ 启发了：[[SnapKV]]（改进为 prefill-time 压缩），[[KVzip]]（改进为 query-agnostic），[[TriAttention]]（改进为 pre-RoPE 评分），[[StreamingLLM]]（简化为 sink + window）
→ 概念关联：[[Token Eviction]]（eviction 方法的代表作），[[KV Cache Management]]（KV cache 管理的子方向之一）
→ 与量化正交互补：可与 [[KIVI]]、[[KVQuant]] 等量化方法组合使用
