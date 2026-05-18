---
title: "Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference"
aliases: [Ada-KV, Ada-SnapKV, Ada-Pyramid, Adaptive Budget Allocation]
source_type: paper
source_url: "https://arxiv.org/abs/2407.11550"
source_date: 2024-07-16
ingested: 2026-05-15
venue: NeurIPS 2025
authors: [Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, S. Kevin Zhou]
affiliation: University of Science and Technology of China (USTC) / MIRACLE Center
code: "https://github.com/FFY0/AdaKV"
tags: [model-compression, kv-cache, eviction, budget-allocation, head-adaptive, theoretical-bound, plug-and-play, inference]
---

## 核心要点

Ada-KV 是首个**head-wise adaptive budget allocation** 策略——它不改变 eviction 方法本身（如 SnapKV、PyramidKV），而是通过理论框架指导如何在不同 attention head 之间最优分配固定总 budget。核心洞察：不同 attention head 的注意力集中度（concentration）差异极大——sparse head 仅需少量 KV 对即可覆盖绝大部分注意力权重，而 dispersed head 需要更大 cache 才能保留重要信息。传统 uniform allocation（每个 head 分配相同 budget）浪费了 sparse head 的配额，同时饿死了 dispersed head。

Ada-KV 的解法极其简洁：**跨所有 head 选取全局 Top-B 的 attention weights，按每个 head 中被选中的数量作为该 head 的 budget**。这等价于将 budget 从 sparse head 转移到 dispersed head。

![[images/Ada-KV_fig1_motivation.png]]
*Figure 1：Left——Llama-3.1-8B 中不同 head 的注意力集中度差异显著。Right——Adaptive allocation 将 budget 从 sparse head 转移到 dispersed head，retained attention weights 从 2.26 提升到 2.48。*

## 理论框架

### L₁ Eviction Loss 定义

设 $y$ 和 $\hat{y}$ 分别为 eviction 前后的 attention 层输出，eviction loss 定义为：

$$L_1 \text{ Eviction Loss} = \|y - \hat{y}\|_1$$

### Theorem 3.1：Eviction Loss 上界

$$L_1 \text{ Eviction Loss} \leq \epsilon = 2hC - 2C \sum_{i \in [1,h]} \sum_{j \in [1,n]} I_i^j A_i^j$$

其中 $C = \max\{\|V_i W_i^O\|_\infty\}$ 是常数（Value 矩阵乘 output projection 的最大行范数），$I_i^j$ 是 eviction decision indicator，$A_i^j$ 是 attention weight。

**关键洞察**：上界 $\epsilon$ 中的可优化项是 $\sum_i \sum_j I_i^j A_i^j$——即被**保留**的 token 的 attention weights 之和。最小化上界 ≡ 最大化保留 token 的总 attention weight。

### Theorem 3.2：Top-k eviction 最小化上界

给定 budget allocation $\{B_i\}$，Top-k eviction decision $\{I_i^*\}$（保留每个 head 中 attention weight 最大的 $B_i$ 个 token）是上界 $\epsilon$ 的最优解：

$$\epsilon^* = \min_{\{I_i\}} \epsilon = 2hC - 2C \sum_{i \in [1,h]} \sum_{A_i^j \in \text{Top-k}(A_i, B_i)} A_i^j$$

这解释了为什么现有的 Top-k eviction 方法（H2O、SnapKV 等）有效——它们实际上在最小化 eviction loss 的上界。

### Theorem 3.3：Adaptive allocation 最小化上界

Ada-KV 的 adaptive budget allocation $\{B_i^*\}$（Algorithm 1）在所有可能的 allocation 中实现最小上界：

$$\epsilon^{**} = \min_{\{B_i\}} \epsilon^* \leq \epsilon^* \text{ for any other } \{B_i\}$$

**证明直觉**：全局 Top-B 选择等价于在所有 head 的 attention weights 中挑选最大的 B 个——这显然比在每个 head 内独立选 Top-$B_i$（local optimum）能保留更大的总 attention weight（global optimum）。

## 算法

### Algorithm 1：Ada-KV

```
Input: Total budget B; Attention weights for head i: {A_i}
Output: Allocated budgets {B_i*}

1. Concatenate attention weights across heads: A = Cat({A_i})
2. Select top B weights from A: Top-k(A, k=B)
3. Count number of selected weights for each head i: {f_i}
4. Set allocated budgets: {B_i* = f_i}

Return {B_i*}
```

**Safeguard 机制**：为避免过于激进的 budget 重分配，实际部署中使用平滑：

$$B_i^* = \alpha \cdot B_i^* + (1-\alpha) \cdot (B/h)$$

其中 $\alpha = 0.2$（固定值，对模型和 budget 不敏感，Table 4 robustness analysis）。

### 与现有方法集成

Ada-KV 通过替换 budget allocation 步骤，plug-and-play 地增强现有 eviction 方法：

- **Ada-SnapKV**：在 SnapKV 的 observation window voting 后，用 Algorithm 1 替代 uniform allocation
- **Ada-Pyramid**：类似地应用于 PyramidKV

集成仅需修改 Algorithm 2 的第 7 步——从 `B/h` 改为 Algorithm 1 的输出。

## GQA 兼容性

论文指出现有方法（SnapKV、PyramidKV）缺乏 GQA 兼容——它们冗余地为每个 query head 复制 KV cache。Ada-KV 实现了 GQA-compatible 机制：用每个 group 内 query head 的**平均 attention weight** 作为选择标准，消除冗余。这对 Llama-3.1-8B（GQA 4:1）带来 4× 额外 cache 压缩。

## 实验设置

- **模型**：Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.2（均使用 GQA）
- **扩展模型**：Llama-3.1-70B（Table 2）
- **Baselines**：SnapKV, Pyramid (PyramidKV), StreamingLLM
- **评估**：
  - Ruler benchmark：13 tasks（NIAH 变体 + CWE + FWE + VT + QA），16K context
  - LongBench：16 datasets，6 task domains
- **场景**：Question-aware（标准）+ Question-agnostic（更贴近实际，question 不参与 cache compression）

## 主要结果

### Ruler Benchmark

![[images/Ada-KV_fig3_ruler.png]]
*Figure 3：Ruler 13 tasks 平均分。Ada-SnapKV/Ada-Pyramid 在所有模型和场景下一致优于原始 SnapKV/Pyramid，尤其在低 budget（20%-40%）下差距更大。*

核心发现：
- **Question-agnostic 场景是真正的挑战**：所有方法在 question-agnostic 下性能大幅下降（Llama 20% budget: SnapKV question-aware ~82 vs question-agnostic ~30），说明 question-aware evaluation 高估了方法的实际能力
- **Ada-KV 在低 budget 下提升最大**：Llama 20% budget question-agnostic, Ada-SnapKV 大幅优于 SnapKV；80% budget 时差距缩小（因为 budget 充裕时 uniform 也足够）
- **子任务分析**（Figure 4）：在 Multi-Key NIAH、Variable Tracking 等需要精确检索的任务中提升最显著

### LongBench Benchmark

![[images/Ada-KV_fig5_longbench.png]]
*Figure 5：LongBench 16 datasets 平均分。固定 budget（128-2048）下，Ada-KV 一致提升 1-3 分。*

![[images/Ada-KV_table1_task_analysis.png]]
*Table 1：Llama-3.1-8B Question-agnostic 按任务域分析。Ada-SnapKV 在 6 大域中全面领先，Few-shot（+2.51/+1.75/+1.33）和 Multi-Doc QA（+2.45/+1.39/+0.86）提升最大。*

- **Question-agnostic 总平均**：Ada-SnapKV 37.45 vs SnapKV 36.38（10% budget）；42.87 vs 41.29（20%）；46.24 vs 45.52（40%）
- **Llama-3.1-70B 上同样有效**（Table 2）：20% budget 46.08 vs 44.95（+1.13）；40% budget 49.64 vs 48.91（+0.73）
- **论文推荐 question-agnostic 作为更真实的评估标准**——future work 应更重视此场景

### 广泛适用性：Follow-up Methods

![[images/Ada-KV_table3_fig6_followup_efficiency.png]]
*Table 2：Llama-3.1-70B 结果 + Table 3：Ada-KV 策略应用于后续方法（CriticalKV、DefensiveKV）一致提升性能。Figure 6：Ada-SnapKV 的 peak memory 和 decoding latency 与原始 SnapKV 相当。*

Table 3 显示 Ada-KV 的广泛影响力：
- **CriticalKV + Ada-KV**：20% budget 43.77 vs 42.99（+0.78）
- **DefensiveKV + Ada-KV**：20% budget **46.68** vs 43.78（+2.90）；40% budget 49.21 vs 47.76（+1.45）
- **Plug-and-play 方法（CriticalKV/DefensiveKV + Ada-KV）超越 training-based 方法**（DuoAttention 39.52, HeadKV 42.64 at 20%）

## 效率

- **Variable-length FlashAttention**：Ada-KV 的非均匀 cache 通过 flattened storage layout + custom CUDA kernel 实现高效计算
- **Zero overhead**：Figure 6 显示 Ada-SnapKV 的 peak memory 和 decoding latency 与 SnapKV 完全一致（budget=1024），两者都远优于 full cache
- **GQA 兼容**：进一步 4× KV cache 压缩（Llama-3.1-8B）

## 理论意义

### 统一理论解释

Ada-KV 的理论框架提供了对整个 Top-k eviction 方法族的统一理解：

1. **为什么 Top-k eviction 有效**（Theorem 3.2）：它在给定 budget 下最小化 eviction loss 上界
2. **Uniform allocation 为何次优**：当 head 间 concentration 差异大时，uniform 的上界远大于 adaptive
3. **Adaptive allocation 的最优性**（Theorem 3.3）：全局 Top-B 选择是所有 allocation 中最优的

### 上界的实际意义

论文通过 Figure 2 实证验证：优化上界确实降低了实际 eviction loss——在 Qasper dataset 上，Ada-KV 在 ~200 个 sample 中几乎全部降低了 relative eviction loss（20% 和 40% budget 下均成立）。

### 局限性

- **上界而非精确值**：降低 upper bound 不严格保证降低实际 loss，但实验一致表明两者高度相关
- **常数 C 的假设**：C = max row norm of $V_i W_i^O$ 在不同 head 间可能差异大，当前理论未考虑 head-wise C 的影响
- **单 query 假设**：理论基于 window size = 1（单个 query state），实际 SnapKV 使用 observation window 内多个 query states

## 研究意义与影响

### 与相关工作的关系

- **vs [[SnapKV]] / [[PyramidKV]]**：Ada-KV 是这些方法的 plug-and-play 增强——不替代它们的 scoring 机制，仅优化 budget 分配
- **vs [[DuoAttention]]**：DuoAttention 通过 training-based profiling 做 head 分类（binary：full vs streaming），Ada-KV 是 training-free 的连续 budget 分配。Table 3 显示 Ada-KV 增强的方法超越 DuoAttention
- **vs [[CAKE]]**：CAKE 做 layer-adaptive allocation（不同层不同 budget），Ada-KV 做 head-adaptive allocation。两者维度不同，理论上可组合
- **vs [[CAOTE]]**：CAOTE 改进 eviction scoring（attention → attention+value），Ada-KV 改进 budget allocation。完全正交，可组合使用
- **vs [[KVzip]]**：KVzip 的 non-uniform head budget 是 training-based 的（通过 importance scoring 获得），Ada-KV 的 allocation 是 inference-time 动态计算的

### 开创性贡献

1. **首个 head-wise adaptive budget allocation**：此前所有方法都使用 uniform allocation
2. **理论框架**：首次建立 Top-k eviction 的 L₁ loss upper bound，解释了现有方法的优化目标
3. **Plug-and-play 设计**：无需修改 eviction 方法本身，仅需在 allocation 步骤插入 Algorithm 1
4. **广泛被 follow-up 采纳**：CriticalKV、DefensiveKV、KVzip、Expected Attention 等后续工作都采用了 Ada-KV 策略

### 对 question-aware vs question-agnostic 的贡献

论文系统地指出 question-aware compression（将 question 包含在 observation window 中）**高估了 eviction 方法的实际能力**。在实际 serving 场景中（如 prefix caching、shared context across queries），cache compression 在 question 到达前就已完成（question-agnostic）。论文建议未来评估应更重视 question-agnostic 场景。

## 引用关系

**本页引用**：
- [[SnapKV]] — Ada-KV 的主要 baseline 和集成目标
- [[PyramidKV]] — 另一个集成目标（Ada-Pyramid）
- [[DuoAttention]] — Training-based head 分类方法，Table 3 对比
- [[CAKE]] — Layer-adaptive allocation，与 Ada-KV 的 head-adaptive 维度正交
- [[CAOTE]] — Scoring 改进（value-aware），与 Ada-KV 的 allocation 改进正交
- [[KVzip]] — Follow-up 工作之一，采纳了 Ada-KV 策略
- [[Expected Attention]] — Follow-up 工作之一，采纳了 Ada-KV 策略
- [[Token Eviction]] — Ada-KV 属于 Token Eviction 子方向（budget allocation 轴）
