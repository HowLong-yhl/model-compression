---
title: "KEYDIFF: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments"
aliases: [KeyDiff, KEYDIFF, Key Diversity Eviction]
source_type: paper
source_url: "https://arxiv.org/abs/2504.15364"
source_date: 2025-06-01
ingested: 2026-05-12
venue: NeurIPS 2025
authors: [Junyoung Park, Dalton Jones, Matthew J Morse, Raghavv Goel, Mingu Lee, Chris Lott]
affiliation: Qualcomm AI Research
code: null
tags: [model-compression, kv-cache, eviction, attention-free, edge-inference, long-context, resource-constrained]
---

## 核心要点

KEYDIFF 是一种 **attention-free** 的 KV cache eviction 方法，核心洞察：**几何上独特的 Key（与其他 Key 余弦相似度低的向量）倾向于获得高注意力分数**。基于这一现象，KEYDIFF 仅通过 Key 向量间的相似度来评估 token 重要性，完全不依赖 attention score。

与现有方法（[[H2O]]、[[SnapKV]]、TOVA）的关键区别：

1. **Attention-free**：不需要 materialize attention weight matrix，可直接配合 FlashAttention 使用
2. **Block prompt processing 友好**：在资源受限环境中分块处理 prompt 时，attention-based 方法因缺少全局上下文而退化，KEYDIFF 的 Key 几何度量天然不受此影响
3. **O(n) 高效计算**：通过 anchor vector 将 O(n²) 的 pairwise 计算降为 O(n) 的单向量比较

结果：Llama-3.1-8B 在 LongBench 上 8K cache budget（~23% 压缩）仅 **≤0.04% 精度损失**；6K cache（~33% 压缩）仅 **≤1.5%** 损失。端到端推理延迟比 TOVA/SnapKV **降低 30%**。

## 动机：Block Prompt Processing 下的 Eviction 难题

### 资源受限环境的推理约束

在边缘设备上部署 LLM 时，内存预算严格受限。传统 eviction 方法（H2O、SnapKV）假设可以一次性处理整个 prompt 后再 eviction，但中间 KV cache 可能已超出内存上限。

解决方案：**Block prompt processing**——将 prompt 分割为大小为 B 的块，逐块处理，每块处理后立即执行 eviction：

```
C_i ← ([K_{i-1} ∥ k_{Bi:B(i+1)-1}], [V_{i-1} ∥ v_{Bi:B(i+1)-1}])
C'_i ← π_N(C_i),  C_i ← C'_i,  C_0 = ∅
```

此策略在整个推理过程中维持严格内存约束。

### Attention-based 方法在 Block Processing 下的退化

Block prompt processing 引入了根本性挑战：当处理 block X_i 时，attention-based eviction 仅能基于 X_0, ..., X_i 的有限上下文评估重要性，无法看到未来 block 中的 query。导致**误 eviction 累积**——早期 block 中被驱逐的 token 可能在后续 block 中至关重要。

实验验证：Table 1 显示所有 attention-based 方法在 block processing 下均有显著退化。

## 核心观察：Key 多样性 ≈ 注意力重要性

KEYDIFF 的理论基石是一个令人惊讶的观察：**Key 向量间的平均成对余弦相似度与 attention score 呈负相关**。

直觉解释：
- Attention score ∝ exp(q·k/√d)，即 Key 与 Query 的内积越大，attention 越高
- 如果某个 Key 与大多数其他 Key 不相似（几何上"突出"），它更可能与 Query 方向对齐
- 这本质上是 [[Attention Sink]] 现象的几何解释——sink token 的 Key 在 Key 空间中是"outlier"

关键优势：Key 之间的几何关系**不依赖于任何 query**，因此在 block processing 中不受未来 token 影响。

## 方法：KEYDIFF

### 基本形式（O(n²)）

对当前 cache 中 n 个 Key 向量，计算 pairwise cosine similarity matrix，保留总相似度最低的 N 个：

```
S = topk(-CosSim(K)·1, N)
K' = gather(K, S),  V' = gather(V, S)
```

其中 CosSim(K) ∈ R^{n×n}，CosSim(K)_{ij} = (k_i · k_j) / (‖k_i‖·‖k_j‖)，1 ∈ R^n。

含义：**保留与其他 Key 最不相似的 N 个 Key**——最大化保留集的多样性。

### 高效形式（O(n)）

将 pairwise 计算简化为每个 Key 与 **anchor vector** 的单次比较：

```
S = topk(-CosSim(μ(K̂), k̂_i), N)
```

其中 μ(K̂) = (1/n)·Σ k̂_i 为归一化 Key 的均值（anchor vector），k̂_i = k_i / ‖k_i‖。

实验发现使用未归一化 Key 的均值 μ(K) 替代 μ(K̂) 精度相当。

**保留条件**：在 mild condition 下，高效形式与完整形式保留相同 KV 对。

### KEYDIFF with Sliding Window

对推理和编码任务（最近 token 通常重要），可将 cache budget 的一部分分配给滑动窗口：

- 固定比例的 budget 保留最近 token（无条件保留）
- 剩余 budget 用 KEYDIFF 评分
- 无额外计算/内存开销，对推理任务有显著提升

## 理论分析

### Theorem 3.1：Key 独特性 → 高 Attention

设 k* 为 cache 中一个 Key，‖k*‖² < M，attention weight w > 0。当 cache 大小 n → ∞ 时：

```
CosSim(k*, q) ≥ -log(1-w) / (2M) - 1
```

即：如果 k* 获得了正的 attention weight w，则 k* 与 query q 的余弦相似度有一个正下界（随 w 增大而增大）。

### Theorem 3.2：Key-Query 对齐 → 低 Key-Mean 相似度

设 CosSim(k*, q) = β_q > 0（Key 与 Query 对齐），CosSim(k̄, q) = α_q < 0（Key 均值与 Query 不对齐）。则：

```
CosSim(k̄, k*) ≤ 1 + α_q·β_q - 0.5·α_q² - 0.5·β_q²
```

### 组合含义

两个定理组合：高 attention weight 的 Key → 与 Query 高度对齐（Thm 3.1）→ 与 Key 均值低相似度（Thm 3.2）→ KEYDIFF 会优先保留。

**KEYDIFF 的最优性**：KEYDIFF 实际上求解了一个**最大化 Key 多样性的子集选择问题**——最小化保留集中 Key 的 pairwise cosine similarity。

## 实验结果

### LongBench（Block size B=128）

| 模型 | 方法 | Cache | Avg Score | vs Full |
|------|------|-------|-----------|---------|
| Llama-3.1-8B | Full (no eviction) | ∞ | 49.20 | — |
| | H2O | 8K | 38.20 | -11.00 |
| | TOVA | 8K | 47.09 | -2.11 |
| | Sink | 8K | 44.68 | -4.52 |
| | SnapKV | 8K | 48.06 | -1.14 |
| | **KEYDIFF** | **8K** | **49.03** | **-0.17** |
| | **KEYDIFF** | **6K** | **48.51** | **-0.69** |
| Llama-3.2-3B | Full (no eviction) | ∞ | 45.87 | — |
| | SnapKV | 8K | 44.96 | -0.91 |
| | **KEYDIFF** | **8K** | **45.78** | **-0.09** |

KEYDIFF 在 8K cache 下几乎无损（-0.17/-0.09），远优于所有 baseline。特别在 **Passage Retrieval** 任务上优势巨大（99.50 vs H2O 62.50），说明 Key 多样性对长程依赖至关重要。

### 2K Budget（极限压缩 ~80%）

KEYDIFF 在 2K budget 下得分 44.33（Llama-3.1-8B），远超所有其他方法（H2O 16.89, TOVA 37.52, SnapKV 39.60），展示了 Key 多样性在极限压缩下的鲁棒性。

### Needle-In-A-Haystack

KEYDIFF 在长文档（>20K tokens）中的 recall 显著优于 TOVA、SnapKV 和 SinkAttention，尤其在 deep needle（信息嵌在文档深处）场景下。

### Math-500 推理

DeepSeek-R1-Distill-Llama-8B + KEYDIFF with sliding window 在 Math-500 上达到近基线水平，优于 SnapKV，证明方法对长生成推理任务也有效。

### 推理延迟

| 方法 | 需要 Attention 矩阵 | 可用 FlashAttention | 相对延迟 |
|------|---------------------|--------------------| ---------|
| TOVA | ✓ | ✗ | 1.30× |
| SnapKV | ✓ | ✗ | 1.30× |
| **KEYDIFF** | **✗** | **✓** | **1.00×** |

由于 KEYDIFF 完全不需要 attention weight，可以直接使用 FlashAttention，TTFT（Time-to-First-Token）比 attention-based 方法**低 30%**。

## 与其他 KV Cache 管理方法的关系

### 与 Attention-based Eviction 的对比

| 维度 | Attention-based (H2O/SnapKV/TOVA) | KEYDIFF |
|------|-----------------------------------|---------|
| 评分信号 | Attention score (query-dependent) | Key cosine similarity (query-free) |
| Block processing 鲁棒性 | 差（仅看当前 block 的 query） | 强（Key 几何不受 query 影响） |
| FlashAttention 兼容 | 否（需 materialize A） | 是 |
| 计算复杂度 | O(n) per head (需 A) | O(n) per head (anchor 比较) |
| 极限压缩鲁棒性 | 退化严重（2K: ~17-40） | 强（2K: 44.33） |

### 与 KeepKV 的互补性

[[KeepKV]] 将 eviction 升级为 merging（合并而非丢弃），解决不可逆信息损失；KEYDIFF 改进 eviction 的**评分方式**（Key 多样性 vs attention score）。两者解决不同维度的问题：
- KEYDIFF 回答"保留哪些 token？"（scoring/selection）
- KeepKV 回答"被选中驱逐的 token 怎么处理？"（eviction vs merging）

理论上可组合：用 KEYDIFF 做 scoring → KeepKV 做 merging（而非直接丢弃）。

### 与 KV Cache 量化的正交性

KEYDIFF 减少 KV cache 的 **token 数量**，量化（[[KIVI]]、[[KVQuant]]、[[KVTuner]]）减少每个 token 的 **比特数**。完全正交，可叠加使用。

### 与 TriAttention 的对比

[[TriAttention]] 也是一种不依赖当前 query 的 eviction 方法（query-agnostic），使用 Pre-RoPE Key 的三角级数特征评分。与 KEYDIFF 的区别：
- TriAttention 利用 Key 的三角结构特征（Pre-RoPE 空间）
- KEYDIFF 利用 Key 的**多样性**（post-RoPE 空间的几何独特性）
- KEYDIFF 专门针对 block processing 场景优化，TriAttention 针对长推理

## 关联页面

- [[Token Eviction]]——KEYDIFF 属于 eviction 方法，改进评分信号为 attention-free
- [[Attention Sparsity]]——KEYDIFF 的理论基础：sparse attention 意味着少数 Key 获得大部分权重
- [[Attention Sink]]——KEYDIFF 从几何角度解释了 sink 现象（sink token 的 Key 是几何 outlier）
- [[KeepKV]]——互补方法：KEYDIFF 改进 scoring，KeepKV 改进 eviction→merging
- [[KV Cache Quantization]]——正交互补：量化降精度 + KEYDIFF 减 token
- [[TriAttention]]——同为 query-agnostic 方法，使用不同评分信号
- [[StreamingLLM]]——KEYDIFF 的 Key 多样性本质上恢复了 attention sink 的保留
