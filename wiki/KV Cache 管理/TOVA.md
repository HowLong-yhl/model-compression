---
title: TOVA
aliases: [Token Omission Via Attention, TOVA Policy]
tags: [model-compression, kv-cache, eviction, attention-sparsity, token-eviction, RNN]
created: 2026-05-12
updated: 2026-05-12
sources: [TOVA]
paper:
  title: "Transformers are Multi-State RNNs"
  authors: [Matanel Oren, Michael Hassid, Nir Yarden, Yossi Adi, Roy Schwartz]
  affiliation: [The Hebrew University of Jerusalem, FAIR (AI at Meta)]
  venue: arXiv 2401.06104 (June 2024)
  url: https://github.com/schwartz-lab-NLP/TOVA
---

## 核心贡献

TOVA（Token Omission Via Attention）是一种 **training-free** 的 KV cache 压缩策略。论文的主要贡献有两层：

1. **理论框架**：将 decoder-only Transformer 重新定义为一种"无界多状态 RNN"（Unbounded Multi-State RNN, MSRNN），其中 KV cache 对应一个动态增长的 multi-state matrix。这为 KV cache 压缩提供了统一的 RNN 视角——所有 eviction 策略都是将无界 MSRNN 转换为有界 MSRNN 的 compression policy
2. **压缩策略**：在每个 decoding step 中，丢弃当前 attention score 最低的 KV state。无需额外训练、无需手工设计 window 或 sink token 规则

## 理论框架：Transformer = 无界 MSRNN

### Multi-State RNN 定义

传统 RNN 维护一个隐状态向量 h_t；MSRNN 将其推广为一个**隐状态矩阵** H_t ∈ R^{g(t)×d}，其中 g(t) 控制状态数量：

- g(t) = 1 → 退化为标准 RNN
- g(t) ≤ k（常数）→ 有界 MSRNN（bounded memory capacity）
- g(t) = t（随时间线性增长）→ 无界 MSRNN

### Transformer 作为无界 MSRNN

Decoder-only Transformer 的自回归推理天然符合 MSRNN 框架：

```
x_{t}^{l+1}, (K_t^l, V_t^l) = f_TRANS(x_t^l, (K_{t-1}^l, V_{t-1}^l))
```

其中 (K_t^l, V_t^l) 就是第 l 层的 multi-state matrix，每步解码增加一行（append 新 token 的 kv）。因为状态矩阵随序列长度线性增长且不丢弃任何历史，Transformer 对应 g(t) = t 的**无界** MSRNN。

### 有界 MSRNN = KV Cache 压缩

将 Transformer 转为有界 MSRNN：设 g(t) = min(t, k)。当 t > k 时需要一个 **compression policy** 来选择保留哪些 state。在这个统一视角下：

| 方法 | 对应的 Compression Policy |
|------|-------------------------|
| Window Attention | FIFO：丢弃最老的 state |
| Window+i | FIFO + 保留前 i 个 state |
| [[H2O]] | 聚合累积 attention score，保留 heavy hitter + window |
| **TOVA** | 每步丢弃当前 attention score 最低的 state |

## TOVA 方法

### 核心机制

每个 decoding step，当 multi-state 超过容量 k 时：

1. 计算当前 query token 对所有 cached KV 的 attention score
2. 找到 attention score 最低的 state j
3. 移除第 j 个 KV 对（将 K_{0:j-1}, K_{j+1:k} 和 V_{0:j-1}, V_{j+1:k} 重新拼接）

### 与 H2O 的关键区别

- **H2O**：聚合**历史所有 query** 的 attention score（累加），保留累积分最高的 token + 固定 window
- **TOVA**：只看**当前 query** 的 attention score，不维护历史累积，不预设 window

TOVA 更简洁——不需要手工决定 window 大小、不需要累积统计。但也因此是 greedy 的。

### 实现细节

- **Head 聚合**：实验表明对同一层内所有 head 的 attention score 取平均后做统一驱逐，优于 head-wise 独立驱逐
- **Training-free**：直接作用于 pretrained LLM，不需要 fine-tuning 或 calibration 数据

## 实验结果

### 语言模型困惑度（PG-19）

在 LLaMA-2-7B、Mistral-7B、Yi-6B 上：

- TOVA 在**所有 multi-state size** 上超越所有 baseline（Window、Window+4、H2O）
- 使用 **1/8 full context**（512 vs 4096）时，PPL 差距仅 0.4 点以内
- 即 512 tokens 的 bounded MSRNN 近似等效于 4096 tokens 的 unbounded 模型

### 长文本理解（SQuALITY / QASPER）

- SQuALITY（摘要生成）：1/4–1/8 full context 即达到 topline 1 分以内
- QASPER（检索 QA）：需要 1/2 full context 才接近 topline——**检索类任务对 cache 大小更敏感**

### 文本生成（GPT-4 评估）

- Multi-state = 1024（1/4 full）：仅 6% 的情况下被 topline 超越
- Multi-state = 256（1/16 full）：47% 的情况下被 topline 超越

### 吞吐量提升

| Multi-state size | KV Memory | Max Batch Size | Relative Throughput |
|-----------------|-----------|---------------|-------------------|
| 256 | 0.15 GB | 139 | 8.5× |
| 512 | 0.28 GB | 70 | 4.8× |
| 1024 | 0.56 GB | 35 | 3.1× |
| 4096 (full) | 2.18 GB | 8 | 1× |

512 tokens 的 TOVA（精度近无损）实现 **4.8× 吞吐提升**。

### 外推能力

使用 position compression（对位置间距取 ln(ln(g))）后，TOVA 可以在 multi-state=512 的条件下**外推到 70K tokens**（训练长度仅 4096），且 PPL 增加 < 0.5。

## 关键发现：哪些 Token 重要

TOVA 的驱逐行为揭示了 token 重要性的分布规律：

### Recency 不是全部

TOVA 保留的 token 中只有 **73-76% 是最近的 token**。其余 24-27% 是较早的 token——说明纯 window 策略会丢失重要信息。

### 第一个 Token 至关重要

第一个 token 在所有 multi-state size 下都被保留到序列末尾——与 [[Attention Sink]]（StreamingLLM）的发现一致。但第 2-25 个 token 被很快丢弃。

### 特定 POS 类型更持久

被保留最久的 token 类型（按 POS tag）：

1. **POS（所有格后缀，如 's）** —— 保留时间最长
2. **引号（"）**
3. **句末标点（. $ ）**
4. **专有名词（NNPS）**
5. **换行符（\n）**

这与 [[H2O]] 发现的"标点 token 是 heavy hitter"一致，但 TOVA 额外发现了**所有格后缀**和**专有名词**的重要性——这些可能编码了实体关系和指代信息。

## 在三轴分类体系中的位置

参照 [[Token Eviction]] 的三轴分类：

| 分类轴 | TOVA 的位置 |
|--------|-----------|
| Query 依赖性 | **Query-Aware**（基于当前 query 的 attention score） |
| Eviction 时机 | **Generation-time**（每个 decoding step 动态驱逐） |
| Budget 分配 | **Fixed (layer-uniform)**（所有层相同 multi-state size） |

### 与其他方法的对比

| 方法 | 需要 Window？ | 需要 Sink？ | 需要历史累积？ | Training-free？ |
|------|:-----------:|:---------:|:-----------:|:-------------:|
| Window+4 | 是（全部） | 是（前 4 个） | 否 | 是 |
| [[H2O]] | 是（一半） | 隐式（累积偏向） | 是 | 是 |
| [[StreamingLLM]] | 是（全部） | 是（前 4 个） | 否 | 是 |
| **TOVA** | **否** | **否（自动发现）** | **否** | **是** |

TOVA 的最大优势是**不做任何先验假设**（不预设 window、不预设 sink），而是让 attention score 自然决定保留谁。实验表明它自动"重新发现"了 window 和 sink 的重要性。

## 局限性

1. **检索类任务需要更大 cache**：QASPER 结果表明，需要精确检索远处信息时 1/8 cache 不够，需要 1/2
2. **Greedy 策略**：只看当前 attention score，不考虑未来 query 可能需要哪些 token（与 H2O 的 local greedy 类似的问题）
3. **仅验证到 13B 规模**：未在更大模型（70B+）或 MoE 模型上验证
4. **单语言**：仅在英文任务上验证，作者指出 word order 更灵活的语言可能需要更大 multi-state

## 引用关系

← 理论基础：[[Attention Sparsity]]（驱逐的可行性前提），[[Attention Sink]]（第一个 token 的重要性，TOVA 自动重现了此发现）
← 方法对比：[[H2O]]（累积 attention eviction），[[StreamingLLM]]（sink + window），[[Scissorhands]]（pivotal token persistence）
← 框架层面：[[Token Eviction]]（三轴分类体系），[[KV Cache Management]]（四大子方向之一）
→ 后续工作：PyramidKV（层自适应 budget）、KVzip（query-agnostic 路线）
