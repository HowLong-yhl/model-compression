---
title: "Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution"
aliases: [Expected Attention, EA, KVPress]
source_type: paper
source_url: "https://arxiv.org/abs/2505.00000"
source_date: 2025-06-01
ingested: 2026-05-12
venue: arXiv 2025
authors: [Alessio Devoto, Maximilian Jeblick, Simon Jégou]
affiliation: Sapienza University of Rome / NVIDIA
code: "https://github.com/NVIDIA/kvpress"
tags: [model-compression, kv-cache, eviction, attention-prediction, gaussian, training-free, prefill, decoding, long-context]
---

## 核心要点

Expected Attention 是一种 **training-free** 的 KV cache 压缩方法，核心创新在于：不依赖过去的 attention score（不可获取于 FlashAttention），也不使用启发式度量，而是利用 LLM hidden state 的**高斯分布性质**，以**闭合形式**计算每个 KV 对对**未来 query** 的期望注意力贡献。

关键优势：

1. **双阶段通用**：同时适用于 prefilling（一次性压缩 prompt KV）和 decoding（流式压缩生成 KV），现有方法通常只针对其一（SnapKV → prefill，StreamingLLM → decode）
2. **理论驱动**：基于 Gaussian MGF（矩生成函数）精确计算期望 attention score，而非启发式
3. **FlashAttention 兼容**：不需要 materialize attention matrix，不需要过去的 attention score
4. **Residual stream 视角**：重要性度量同时考虑 attention weight 和 value 影响力

结果：Qwen3-8B 在 Ruler 4K 上 50% 压缩保持 94.7（vs baseline 95.3），90% 压缩仍达 65.4；MATH-500 decoding 12× 压缩下显著优于所有 baseline。同时发布 **KVPress** 库（含 20+ 压缩方法的统一基准框架）。

## 动机：预测未来 Attention 的挑战

### KV 对的重要性本质

每个 KV 对 (k_i, v_i) 对 token t 的输出贡献为 residual stream 中的增量：

```
Δh_{ti} = a_{ti} · W_o · v_i
‖Δh_{ti}‖ = a_{ti} · ‖W_o v_i‖
```

最优的 importance score 应综合两个因素：
- **a_{ti}**（attention weight）：query 对该 Key 的关注度
- **‖W_o v_i‖**（transformed value magnitude）：该 Value 对输出的影响力

### 现有方法的根本困境

| 困境 | 说明 | 受影响方法 |
|------|------|-----------|
| **未来 attention 不可知** | 压缩时不知道未来 query 会 attend 到哪些 KV | 所有 attention-based 方法 |
| **FlashAttention 不 materialize A** | 现代推理实现不保存完整 attention matrix | TOVA, SnapKV, H2O |
| **仅适用单阶段** | Prefill-only 或 decode-only，无法通用 | SnapKV (prefill), StreamingLLM (decode) |

Expected Attention 通过**从 query 分布预测未来 attention**来一并解决这三个问题。

## 方法：Expected Attention

### Step 1：Hidden State 的高斯假设

**关键观察**：LLM 中间层的 hidden state 服从近似高斯分布 h ~ N(μ, Σ)。

这一性质经多个模型架构验证（Llama、Qwen、Gemma），见论文 Figure 1 和 Section B。

![[images/EA_fig1_gaussian_hidden_states.png]]
*Figure 1：Llama3.1-8B 中 Layer 16/20 的 hidden state 分布（红色）和 Layer 20 对应 query 分布（蓝色），叠加最佳高斯拟合（虚线）。Hidden state 和 query 均近似正态，为 Expected Attention 的 MGF 推导提供了理论前提。*

### Step 2：Query 分布的推导

由于 query 是 hidden state 的线性变换（加 RoPE），query 也继承高斯性质：

```
q_t = R_t · W_Q · h_t  →  q_t ~ N(μ_{q_t}, Σ_{q_t})
其中：μ_{q_t} = R_t · W_Q · μ,  Σ_{q_t} = R_t · W_Q · Σ · W_Q^T · R_t^T
```

### Step 3：位置平均化

为创建对未来一段区间的统一表示，对未来 T 个位置的 RoPE 矩阵取平均：

```
R̄ = (1/T) · Σ_{j=1}^{T} R_{t+j}
q̄ ~ N(μ̄_q, Σ̄_q),  其中 μ̄_q = R̄·W_Q·μ,  Σ̄_q = R̄·W_Q·Σ·W_Q^T·R̄^T
```

实验中使用 T=512。

### Step 4：期望注意力分数（闭合形式）

对高斯分布的 query q̄ 和固定的 key k_i，利用**矩生成函数**直接计算期望 unnormalized attention：

```
ẑ_i = E_{q̄~N(μ̄_q, Σ̄_q)} [exp(q̄ᵀk_i / √d)]
    = exp(μ̄_qᵀk_i / √d  +  k_iᵀΣ̄_q k_i / (2d))
```

第一项是均值 query 与 key 的 attention（信号项），第二项是 query 方差对该 key 方向的贡献（不确定性项）。

归一化得到期望 attention weight：

```
â_i = ẑ_i / Σ_{j=1}^{t} ẑ_j
```

### Step 5：综合重要性评分

结合 attention 预测和 value 影响力：

```
‖Δ̂h_i‖ = (â_i + ε) · ‖W_o v_i‖
```

其中 ε=0.02 是正则化超参数（避免完全忽略低 attention 的 KV 对）。

### 压缩算法

按 ‖Δ̂h_i‖ 排序，移除得分最低的 r% KV 对。支持 head-adaptive 压缩（不同 head 保留不同数量）。

### 伪代码

```python
def compress(queries, keys, values, compression_ratio):
    # 计算 query 统计量
    mean_query, cov_query = compute_statistics(queries)
    # 计算 unnormalized attention scores
    scores = matmul(mean_query, keys.T) / sqrt(d)
    scores += einsum("i,ij,j->", keys, cov_query, keys) / (2 * d)
    # 归一化并加权 value norms
    scores = softmax(scores) * values.norm(dim=-1)
    # 保留最高分的 KV 对
    n_kept = int(keys.size(0) * (1 - compression_ratio))
    indices = scores.topk(n_kept).indices
    return keys[indices], values[indices]
```

## 实验结果

### Ruler（长上下文检索/推理/聚合）

| 模型 | 方法 | 0% | 10% | 25% | 50% | 75% | 90% |
|------|------|-----|-----|-----|-----|-----|-----|
| Qwen3-8B | **EA** | 95.3 | **95.3** | **95.0** | **94.7** | **88.3** | **65.4** |
| | TOVA | 95.3 | 89.0 | 82.5 | 77.6 | 62.4 | 24.7 |
| | SnapKV | 95.3 | 92.6 | 84.0 | 55.7 | 33.1 | 19.2 |
| | KeyDiff | 95.3 | 93.8 | 89.4 | 78.6 | 64.4 | 37.9 |
| Gemma3-12B | **EA** | 95.2 | **95.2** | **94.9** | **92.7** | **78.2** | **53.6** |
| | KeyDiff | 95.2 | 94.3 | 90.6 | 79.8 | 62.0 | 34.3 |
| Llama3.1-8B | **EA** | 95.3 | **95.7** | **95.3** | 92.2 | 75.9 | 30.6 |
| | KeyDiff | 95.3 | 94.7 | 91.6 | 85.5 | **72.9** | **61.1** |

**关键观察**：EA 在 Qwen3 和 Gemma3 上全面领先；在 Llama3.1 上 50% 以内领先，但 75%+ 高压缩下 KeyDiff 更鲁棒（可能因 Llama 的 hidden state 分布偏离高斯假设较多）。

### LongBench（多任务长上下文）

EA 在 Qwen3-8B 和 Gemma3-12B 上跨所有压缩比实现最优 trade-off：在大多数数据集上，EA 的压缩-性能曲线始终处于 Pareto 最优位置。尤其在 Passage Retrieval 任务上，高压缩比下 EA 的优势最为明显。

![[images/EA_fig2_longbench_curves.png]]
*Figure 2：Qwen3-8B（上）和 Gemma3-12B（下）在 LongBench 各子集上的压缩比-性能曲线。x 轴为压缩比，y 轴为任务得分。Expected Attention（红线）在几乎所有数据集和压缩比下均处于 Pareto 最优位置。*

### MATH-500 Decoding（推理模型生成压缩）

| 模型 | 方法 | 0× | 2× | 4× | 12× |
|------|------|-----|-----|-----|------|
| Qwen-R1-7B | **EA** | 0.57 | **0.55** | **0.53** | **0.49** |
| | KeyDiff | 0.57 | 0.54 | 0.48 | 0.35 |
| | StreamingLLM | 0.57 | 0.54 | 0.51 | 0.41 |
| | KNorm | 0.57 | 0.47 | 0.32 | 0.12 |
| Nemotron-14B | **EA** | 0.57 | **0.55** | **0.54** | **0.47** |
| | KeyDiff | 0.57 | 0.56 | 0.51 | 0.44 |
| | StreamingLLM | 0.57 | 0.57 | 0.54 | 0.42 |

EA 在高压缩场景（12×）下优势最大——说明推理模型的 thinking trace 确实含有大量可安全剪除的冗余信息。

### Needle-In-A-Haystack

50% 压缩下，EA 在 Llama3.1-8B 上维持完整的 NIAH recall（全绿），优于 TOVA、SnapKV、StreamingLLM、KNorm。

![[images/EA_fig3_needle_in_haystack.png]]
*Figure 3：Llama3.1-8B 在 50% 压缩下的 Needle-in-Haystack 测试。每个热力图的 x 轴为 context length，y 轴为 needle depth。绿色=完美 recall，红色=失败。Expected Attention 和 DuoAttention 保持近乎完美的检索能力，而 TOVA、SnapKV、StreamingLLM 在长文档深处严重退化。*

### 内存效率

- 120K context + 50% compression：峰值内存从 ~43GB 降至 ~28GB（**节省 15GB**）
- 50% 压缩在 Qwen3-8B NIAH 上无精度损失

![[images/EA_fig5_memory_savings.png]]
*Figure 5：(a) Llama3.1-8B 峰值内存随序列长度变化——50% 和 90% EA 压缩 vs 无压缩基线。长上下文（120K）下节省可达 15GB。(b) Qwen3-8B NIAH 得分 vs 压缩比——50% 压缩时零精度损失（气泡大小表示实际 KV cache 大小）。*

## 理论意义

### Gaussian Hidden State 假设的合理性

论文通过多个模型架构广泛验证了 hidden state 的高斯性：各层 hidden state 的边际分布接近正态（Figure 1），query 作为线性变换后的结果保持高斯性。

### MGF 的精妙应用

期望 attention score 的推导利用了高斯随机变量的矩生成函数（MGF）：

```
E[exp(aᵀx)] = exp(aᵀμ + aᵀΣa/2),  x ~ N(μ, Σ)
```

这使得原本需要 Monte Carlo 采样或积分的期望计算变为 O(d²) 的矩阵运算。

### 两项分解的直觉

期望 unnormalized score 中的两项有清晰的物理含义：
- **μ̄_qᵀk_i/√d**：均值 query 对 key_i 的 attention（确定性信号）
- **k_iᵀΣ̄_q k_i/(2d)**：query 分布在 key_i 方向的方差贡献（不确定性补偿）

第二项保证了即使某个 key 不被"平均 query"关注，只要它位于 query 分布方差大的方向，仍被保留——这类似于一种"保险机制"。

## KVPress 库

论文同时发布了 **KVPress**——基于 PyTorch 和 HuggingFace Transformers 的 KV cache 压缩研究框架：

- **20+ 方法统一实现**：Expected Attention, SnapKV, TOVA, KeyDiff, StreamingLLM, KNorm, H2O, DuoAttention, AdaKV, PyramidKV 等
- **非侵入式设计**：通过 PyTorch forward hooks 在每层 attention 后触发压缩，无需修改模型架构
- **标准化评估**：配套 leaderboard 和 benchmark suite
- **研究导向**：优先可读性而非运行效率，便于方法对比和创新

## 与其他 KV Cache 管理方法的关系

### 方法论定位

| 方法 | 评分信号 | Prefill | Decode | 理论基础 | FlashAttn 兼容 |
|------|---------|---------|--------|----------|--------------|
| SnapKV | 过去 query 的 attention | ✓ | ✗ | 启发式 | ✗ |
| TOVA | 累积 attention score | ✓ | ✓ | 启发式 | ✗ |
| StreamingLLM | 位置（sink + window） | ✗ | ✓ | Attention Sink 观察 | ✓ |
| KNorm | Key 的 L2 范数 | ✗ | ✓ | 启发式 | ✓ |
| [[KeyDiff]] | Key 几何多样性 | ✓ | ✓ | 多样性最优化 | ✓ |
| [[KeepKV]] | 任意 scorer → merging | ✓ | ✓ | ZIP-Merging 零扰动 | — |
| **Expected Attention** | **Gaussian MGF 预测未来 attention** | **✓** | **✓** | **分布论 + MGF** | **✓** |

### 与 KeyDiff 的对比

两者都是 FlashAttention-compatible、双阶段通用的方法，但思路截然不同：
- **KeyDiff**：Key 的**几何多样性**作为重要性代理（保留独特 Key）→ 基于当前状态的静态属性
- **Expected Attention**：**预测未来 query** 对每个 Key 的期望 attention → 基于对未来行为的分布估计

实验对比：EA 在 Qwen3/Gemma3 上全面优于 KeyDiff；但 KeyDiff 在 Llama3.1 的极高压缩（75-90%）下更鲁棒。可能原因：Llama 的 hidden state 与高斯假设偏差较大时，MGF 近似退化；而 KeyDiff 的几何度量不依赖分布假设。

### 与 KV Cache 量化的正交性

Expected Attention 减少 KV cache 的 **token 数量**（eviction），量化（[[KIVI]]、[[KVQuant]]）减少每个 token 的 **比特数**。完全正交，可叠加。论文明确提到量化方法与 KV cache compression "orthogonal, making it possible to integrate them"。

### 与 KeepKV 的组合可能

Expected Attention 可作为 [[KeepKV]] 的 **scorer**——用 EA 评分决定哪些 token 要被移除，然后用 ZIP-Merging（而非直接丢弃）处理低分 token，实现更温和的压缩。

## 局限性

1. **不如可训练方法**：DuoAttention 等方法通过学习压缩 mask 可以更精确，但需要额外训练
2. **压缩比需手动指定**：缺乏自动确定最优压缩率的机制
3. **高斯假设的局限**：某些模型/层的 hidden state 可能偏离高斯（如 Llama3.1 在高压缩下表现不如 KeyDiff）
4. **非优化实现**：当前 PyTorch 实现未经 CUDA kernel 优化，实际部署需进一步工程化

## 关联页面

- [[Token Eviction]]——Expected Attention 属于 eviction 方法，创新在评分机制（分布预测 vs 启发式）
- [[KeyDiff]]——同为 FlashAttention 兼容的双阶段方法，互为竞品/互补
- [[Attention Sparsity]]——EA 隐含利用了 attention 稀疏性：大部分 KV 对的期望 attention 很低
- [[KV Cache Quantization]]——正交互补（减精度 vs 减 token 数）
- [[KeepKV]]——EA 可作为 KeepKV 的 scorer 使用
- [[SnapKV]]——EA 的直接对标方法（SnapKV 用过去 query 的 attention，EA 用未来 query 的期望）
- [[Emergent Outlier Features]]——hidden state 分布性质是 EA 理论基础的前提
