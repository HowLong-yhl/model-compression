---
title: "TriAttention: Efficient Long Reasoning with Trigonometric KV Compression"
aliases: [TriAttention, Mao 2026]
source_type: paper
source_url: "https://arxiv.org/abs/2604.01921"
source_date: 2026-04-06
ingested: 2026-05-14
venue: arXiv 2026
authors: [Weian Mao, Xi Lin, Wei Huang, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, Yukang Chen]
affiliation: MIT, ZJU, NVIDIA
code: "https://github.com/WeianMao/triattention"
tags: [model-compression, method, kv-cache, eviction, reasoning, long-context, pre-RoPE, inference]
---

## 核心要点

TriAttention 专注于**长推理链**（CoT 数万 token）场景下的 KV cache 压缩。核心发现：在 pre-RoPE 空间中，Q 和 K 向量高度**方向集中**（concentration），围绕固定的非零中心分布且跨位置/跨上下文稳定。利用这一性质，attention logit 可以近似为关于 Q-K 距离的**三角级数**（trigonometric series），从而无需实际 query 即可预测哪些 Key token 将获得高 attention。

与 [[SnapKV]] 等基于 post-RoPE attention 的方法不同，TriAttention 的评分完全基于 pre-RoPE 统计量（可离线 calibration），是一种 **query-agnostic** 的 eviction 方法。在长推理任务上以 2.5× 吞吐 / 10.7× KV 内存压缩匹配 Full Attention 精度，大幅领先所有基线。
![[images/Pasted image 20260514203056.png]]
*Figure 1. AIME25 (Qwen3-8B) 上的精度-效率权衡。(A) 等精度 (40.8%) 下 TriAttention 2.5× 吞吐; (B) 10.7× KV 内存压缩。*

## 核心发现：Q/K Concentration

### Pre-RoPE 空间的方向集中性

**现象**：在 pre-RoPE 空间中，Q 和 K 向量不是随机分布的——它们高度集中在固定的非零方向中心 $\bar{q}_f$、$\bar{k}_f$ 附近。用方向统计学中的 **Mean Resultant Length (MRL)** 度量：

$$R_f = \frac{\|\mathbb{E}[q_f]\|}{\mathbb{E}[\|q_f\|]}$$

- $R \to 1$: 完美集中（所有向量同方向）
- $R \to 0$: 均匀分散

约 **90% 的 attention head** 的 R > 0.95，表明极强的方向集中性。这一性质跨位置、跨上下文、跨数据域（Math/Coding/Chat 均为 0.977-0.980）稳定，是**模型内禀属性**。

![[images/TriAttention_fig2_QK_concentration.png]]
*Figure 2. Q/K concentration 及其对 attention 的影响。(A) Pre-RoPE Q/K 在主频率带高度集中 (R≈1.00); (B) RoPE 旋转后变为弧形分散; (C) 绝大多数 head R>0.9; (D) 三角级数重建 attention 的 Pearson r=0.72。*

### 为什么 Post-RoPE 方法受限

现有 eviction 方法（[[H2O]]、[[SnapKV]]、Scissorhands、R-KV）用 post-RoPE query 的 attention 模式来评估 Key 重要性。但 RoPE 使 query 随位置旋转，导致：

1. **Observation window 极小**：仅最近 ~25 个 token 的 query 方向是最新的——扩大窗口反而性能下降（R-KV 论文确认）
2. **Retrieval heads 的致命缺陷**：关键 token 可能在短窗口内获得低 attention 而被永久驱逐，实际它们在后续推理中至关重要
3. **Norm-based 方法同样受限**：只利用向量幅度，忽略方向信息；而 post-RoPE 空间中方向随位置持续变化，无法有效利用

### Trigonometric Series 近似

当 Q/K 集中时，用中心值 $\bar{q}_f$, $\bar{k}_f$ 代入 RoPE attention 公式，logit 简化为距离 Δ 的确定性函数：

$$\text{logit}(\Delta) \approx \sum_f \|\bar{q}_f\| \|\bar{k}_f\| \cos(\omega_f \cdot \Delta + \bar{\phi}_f)$$

其中 $\omega_f = \theta^{-2f/d}$ 是 RoPE 频率，$\bar{\phi}_f = \arg(\bar{q}_f) - \arg(\bar{k}_f)$ 是中心相位差。这是一个**仅依赖距离 Δ 的三角级数**——类似 Fourier 合成（但频率为几何级数而非调和级数）。不同的 Q/K 中心产生不同的 attention-vs-distance 曲线，形成可预测的"距离偏好"：某些特定距离的 token 天然获得更高 attention。

### 实验验证

在三个 DeepSeek-R1 蒸馏模型（Qwen3-8B, DS-Qwen-7B, DS-Llama-8B）上测试三角级数重建实际 attention 的 Pearson 相关性 $\bar{r}$：

![[images/TriAttention_fig3_reconstruction_correlation.png]]
*Figure 3. 三个模型的 per-head 重建相关性分布。均值分别为 0.53、0.56、0.51，分布明显右偏（大量 head 高于 0.6）。*

高相关性跨多 head、多架构成立，确认三角级数从 Q/K 中心准确预测 attention 模式。且验证 Q/K concentration 是 model-intrinsic property：在 Qwen3-8B 上，Math/Coding/Chat 三个域的 MRL 几乎一致 (0.977–0.980)，同样适用于 MLA 架构。

## 方法详解

![[images/TriAttention_fig4_method_overview.png]]
*Figure 4. 方法总览。左：离线 calibration 计算 Q 分布中心 $\mathbb{E}[q_f]$; 中：推理时结合 $S_\text{trig}$（距离偏好）和 $S_\text{norm}$（幅度信号）评分; 右：pruning 后的 attention map。*

### 重要性评分

对每个 cached Key token $k$，评分由两个互补组件组成：

**1. Trigonometric Series Score $S_{\text{trig}}(k, \Delta)$**

使用离线 calibration 得到的 Q 中心 $\mathbb{E}[q_f]$，结合实际 Key 向量 $k_f$，计算在距离 $\Delta$ 处的预期 attention：

$$S_{\text{trig}}(k, \Delta) = \sum_f \|\mathbb{E}[q_f]\| \cdot \|k_f\| \cdot \cos(\omega_f \Delta + \phi_f)$$

其中 $\phi_f = \arg(\mathbb{E}[q_f]) - \arg(k_f)$ 是 Q 中心与当前 Key 的角度差。

**2. Norm-Based Score $S_{\text{norm}}(k)$**

对 concentration 较弱的频率带，norm 提供补充信号：

$$S_{\text{norm}}(k) = \sum_f (1 - R_f) \cdot \mathbb{E}[\|q_f\|] \cdot \|k_f\|$$

等价于 $S_{\text{norm}}(k) = \sum_f (\mathbb{E}[\|q_f\|] - \|\mathbb{E}[q_f]\|) \cdot \|k_f\|$——当 $R_f$ 高（集中度强），$(1-R_f)$ 很小，norm 贡献少；当 $R_f$ 低，norm 权重大。

**3. Combined Score**

$$S(k, \Delta) = S_{\text{trig}}(k, \Delta) + S_{\text{norm}}(k)$$

$R_f$ 自然充当自适应权重，无需手动调参。

**4. Future Offset Averaging**

Key 可能被任何未来位置查询，因此在多个未来偏移 $D = \{1, 2, 4, ..., 2^{16}\}$（几何间距，共 17 个点）上平均评分：

$$\tilde{S}(k) = \frac{1}{|D|} \sum_{\delta \in D} S(k, \Delta + \delta)$$

几何间距至关重要：近距离需要更密采样（线性间距精度 -17.1%）。最大距离从 128 扩展到 4096 带来 +7.1% 精度提升。

### GQA 支持

在 Grouped-Query Attention 中，G 个 query head 共享一个 KV head，各 head 评分尺度不同。采用 **normalize-then-aggregate** 策略：

$$S_{\text{final}}(k) = \max_{g \in \{0,...,G-1\}} \hat{S}^{(g)}(k)$$

其中 $\hat{S}^{(g)}(k)$ 是第 $g$ 个 query head 的 z-score 归一化评分。Max 聚合确保任何一个 query head 认为重要的 Key 都被保留。

### 执行策略

- **Window-based Pruning**：每累积 $\beta = 128$ 个新 token 触发一次 eviction（与 R-KV 一致），超过 budget B 时评分全部 Key，保留 top-B
- **Offline Calibration**：一次性在少量数据上计算 $\mathbb{E}[q_f]$ 和 $\mathbb{E}[\|q_f\|]$（50K-960K token 均可，对数据量/质量不敏感——用 Google homepage HTML 和 ShareGPT 结果几乎相同）
- **跨域泛化**：在 coding 数据上 calibration → 推理任务上精度几乎不退化

## 实验结果

### 推理任务主结果

评估 4 个推理模型：Qwen3-8B、DeepSeek-R1-Distill-Llama-8B、DeepSeek-R1-Distill-Qwen-7B、GPT-OSS-20B。生成长度 32K token。

![[images/TriAttention_table1_reasoning.png]]
*Table 1. AIME24 / AIME25 推理精度。TriAttention 在所有模型上均为最佳 KV 压缩方法，大幅领先 SnapKV 和 R-KV。*

关键数据点（KV budget = 2048）：
- **AIME25 Qwen3-8B**: TriAttention 32.9% vs R-KV 17.5% vs SnapKV 20.0% (Full: 40.8%)——领先 R-KV **+15.4pp**
- **AIME24 Qwen3-8B**: TriAttention 42.1% vs R-KV 25.4% vs SnapKV 34.6% (Full: 57.1%)
- **GPT-OSS-20B AIME24**: TriAttention 59.2% vs R-KV 49.6% (Full: 69.2%)

MATH500 (budget=512): TriAttention 56.0% vs R-KV 46.4% vs SnapKV 49.2% (Full: 69.6%)。Budget 提高到 1024 时 TriAttention 68.4% ≈ Full 69.6%。

### Accuracy vs Budget 全景

![[images/TriAttention_fig5_accuracy_vs_budget.png]]
*Figure 5. (A-C) 三个推理基准上精度随 KV budget 变化。TriAttention 在所有 budget 水平均优于 R-KV，低/中 budget 优势最显著。(D) Memory Retention 基准：DFS 递归深度测试。*

关键发现：
- Budget=4096 时 TriAttention 在 AIME25 达 43.3%——**超过** Full Attention (40.8%)，说明 Full Attention 的 KV cache 本身存在冗余
- Budget=1024 时 MATH500 几乎无损 (68.4% vs 69.6%)

### Memory Retention（DFS 递归测试）

递归算法天然测试记忆保持：需要 descend through nested calls → 保留中间状态 → backtrack。更深的递归 = 更大内存压力。

结果（Qwen3-8B, budget=2048）：
- TriAttention 在深度 ≤16 与 Full Attention 接近，深度 8/12 甚至略优
- R-KV 从深度 14 (61%) **灾难性退化**至深度 16 (31%)——表明其 pruning 策略错误丢弃了关键的中间推理状态
- 只有深度 >18 时 TriAttention 才开始落后于 Full Attention

### 系统性能

| 指标 | Full Attention | TriAttention | 加速比 |
|------|:---:|:---:|:---:|
| MATH500 吞吐 (tok/s) | 222.8 | 1405.2 | **6.3×** |
| AIME25 吞吐 (tok/s) | 222.8 | 563.5 | **2.5×** |
| AIME24 吞吐 (tok/s) | 222.8 | 413.9 | **1.9×** |

与 R-KV 对比（等精度条件）：TriAttention 仅需**一半 KV budget** (1024 vs 2048)，同时吞吐高 85% (1405 vs 760 tok/s)。在等 memory 条件下，MATH500 精度 +8pp，AIME24 精度 +15.4pp。

TriAttention 使 OpenClaw（开源推理模型）可以在**单张消费级 GPU** 上部署长上下文推理——否则 Full Attention 会 OOM。

## 消融实验

| 消融 | AIME24 | AIME25 | 变化 |
|------|:---:|:---:|:---:|
| Full TriAttention | 42.1 | 32.9 | — |
| w/o $S_\text{trig}$（仅 norm） | 18.8 | 21.2 | **-23.3 / -11.7** |
| w/o $R$ 自适应加权 | 41.3 | 28.7 | -0.8 / -4.2 |
| Calibration on Coding (测推理) | 44.2 | 29.2 | +2.1 / -3.7 |
| Calibration on Reasoning | 42.1 | 32.9 | baseline |

**关键消融结论**：
1. **$S_\text{trig}$ 是核心**：移除后精度灾难性下降，确认三角级数距离偏好是本方法的关键信号
2. **$S_\text{norm}$ 有补充价值**：仅用 $S_\text{trig}$（无 norm）比完整方法低 ~5%，说明 norm 对识别低幅值 token（如 attention sink）有用
3. **$R_f$ 自适应加权有效**：去除后 AIME25 -4.2pp
4. **跨域鲁棒**：coding 数据 calibration 用于推理任务，精度几乎不变
5. **Calibration 数据量/质量不敏感**：50K-960K token 结果稳定；Google homepage HTML (低质量) = ShareGPT chat (高质量)

## 局限性

- **Concentration 假设**：~10% 的 head $R < 0.95$，三角级数近似对这些 head 不够准确（用 norm-based 评分补充，但仍是近似）
- **距离偏好是近似**：用中心值替代实际向量，对离中心较远的 token（outlier Q/K）可能不准确
- **仍是 eviction（不可恢复）**：被驱逐的 KV 永久丢失，极深递归（>18 层）时可能丢失关键状态
- **无专用 kernel**：当前未做硬件级优化，进一步加速的 inference kernel 是未来方向
- **未测试 coding/agentic 任务**：主要验证在数学推理上，其他长推理场景的泛化待确认

## 与已有知识的关联

### Pre-RoPE 发现的第三个视角

TriAttention 的 pre-RoPE Q/K concentration 是继 [[KVQuant]]（pre-RoPE Key 通道稳定性）和 [[ShadowKV]]（pre-RoPE Key 低秩性）之后，对"RoPE 前 K 具有更强结构化特性"的第三个独立验证。三者从不同角度发现了同一基础事实：

| 工作 | 发现的 Pre-RoPE 结构 | 利用方式 |
|------|---------------------|---------|
| [[KVQuant]] | 通道稳定性（固定 outlier channel） | Per-channel 量化 |
| [[ShadowKV]] | 低秩性（SVD 6× 压缩） | 低秩投影存储 |
| **TriAttention** | 方向集中性（R > 0.95） | 三角级数 importance 预测 |

### 与现有 Eviction 方法的定位对比

| 方法 | 评估空间 | 信号来源 | 时间依赖 | 长推理适配 |
|------|---------|---------|---------|-----------|
| [[SnapKV]] | post-RoPE | Observation window attention | 需要 recent queries | 窗口外 token 不可恢复 |
| [[H2O]] | post-RoPE | 累积 attention score | 逐步更新 | Heavy Hitter 假设可能在推理中失效 |
| [[DuoAttention]] | post-RoPE | Head-level 二分类 | 固定分类 | Retrieval head 全 KV 保留 |
| R-KV | post-RoPE | Recent query attention + redundancy | 需要 recent queries | 推理场景 best baseline |
| **TriAttention** | **pre-RoPE** | Q/K 中心 + 三角级数 | **Query-agnostic** | 专为长推理设计 |

### 与旋转量化的理论联系

TriAttention 的三角级数分析本质上是在研究 RoPE 旋转如何影响 attention 分数——这与 [[Rotation-based Quantization]] 研究正交旋转如何影响量化误差在数学形式上有相似之处。两者都利用了正交变换的结构化特性。

### 与 MixKV 的互补关系

[[MixKV]] 从 diversity 角度改进 importance-only 方法，TriAttention 则从 pre-RoPE 理论角度彻底绕开 post-RoPE 的 observation window 问题。两者在不同维度解决 eviction 的不同局限，理论上可以结合：用三角级数做初筛 + diversity 做 tie-breaking。

## 引用关系

← 改进自：[[SnapKV]]（从 post-RoPE observation window 到 pre-RoPE 理论评分），[[H2O]]（从经验性 attention score 到理论性三角级数），R-KV（长推理 baseline）
← 理论基础：RoPE (Rotary Position Embedding, Su et al. 2024)，方向统计学（Mean Resultant Length, Mardia & Jupp 1999），[[Attention Sparsity]]
← 相关发现：[[KVQuant]]（Pre-RoPE 通道稳定性），[[ShadowKV]]（Pre-RoPE 低秩性），EliteKV（RoPE 频率选择）
← 对比方法：LazyEviction（延迟驱逐），VATP（value norm 补充 attention score）
→ 贡献：首次将 pre-RoPE Q/K 方向集中性与三角级数理论结合，实现 query-agnostic 理论驱动的 KV 重要性评估
→ 应用场景：长推理链 KV cache 压缩，推理模型部署（单 GPU 长上下文）
→ 概念关联：[[Token Eviction]]（query-agnostic eviction），[[KV Cache Management]]，[[DuoAttention]]（同一 Han Lab）
