---
title: "CAKE: Cascading and Adaptive KV Cache Eviction with Layer Preferences"
aliases: [CAKE, CakeKV, Preference-Prioritized Adaptive Allocation, P2A]
source_type: paper
source_url: "https://openreview.net/forum?id=CAKE_ICLR2025"
source_date: 2025-01-23
ingested: 2026-05-12
venue: ICLR 2025
authors: [Ziran Qin, Yuchen Cao, Mingbao Lin, Wen Hu, Shixuan Fan, Ke Cheng, Weiyao Lin, Jianguo Li]
affiliation: Shanghai Jiao Tong University / Ant Group
code: "https://github.com/antgroup/cakekv"
tags: [model-compression, kv-cache, eviction, layer-adaptive, attention-dynamics, non-uniform-allocation, cascading, long-context]
---

## 核心要点

CAKE 将 KV cache 的预算分配问题比作 **"切蛋糕"（cake-slicing）**：不同 layer 因注意力动态差异，需要不同大小的 cache 预算。核心创新在于定义了两个注意力特征指标（空间 dispersion + 时间 shift），据此自适应地为每层分配不等的 cache 容量，并在 prefilling 过程中以 **cascading（级联）** 方式动态管理缓存。

![[images/CAKE_fig1_attention_patterns.png]]

关键贡献：

1. **Preference-Prioritized Adaptive Allocation (P2A)**：基于 layer 级别的注意力空间分散度和时间变化量，自适应分配 cache budget，取代 uniform 或固定 pyramid 模式
2. **Cascading Cache Management**：prefilling 期间分 L 个 stage 逐层处理，每加入新 layer 即全局重新分配预算并执行 eviction，避免峰值内存超出预算
3. **Attention-Shift Tolerant Eviction Indicator**：eviction 时不仅考虑 mean attention（持续重要性），还加入 variance（注意力波动性），防止过早淘汰"间歇性重要"的 token

结果：仅用 **3.2% KV cache**（128L budget，32K context）即可保持模型性能，128K context 下实现 **10× decoding 加速**和 **~48.6% 峰值内存降低**。在 LongBench 16 个数据集和 NeedleBench 上一致优于 StreamingLLM、H2O、TOVA、SnapKV、PyramidKV。

## 动机：Layer 间的注意力异质性

### 为什么 uniform allocation 不够？

![[images/CAKE_fig2_allocation_strategies.png]]

现有 KV cache eviction 方法主要采用两种分配策略：

- **Uniform allocation**（StreamingLLM、H2O、SnapKV）：每层分配相同的 cache budget → 无法适应不同 layer 的需求差异
- **Fixed-pattern allocation**（PyramidKV）：人工设定 pyramid 形状（浅层多、深层少）→ 仅在特定模型/场景有效，泛化性差

CAKE 的观察：**不同 layer 的注意力模式差异巨大**——有些 layer 注意力高度分散需要保留更多 context，有些 layer 注意力集中在少量 token 可以大幅裁剪。这种差异还随模型、输入上下文动态变化。

### 两个关键注意力特征

![[images/CAKE_fig3_attention_dynamics.png]]

通过 LongBench 数据集在多个 LLM 上的分析，CAKE 识别出两个决定 layer 缓存需求的注意力特征：

**1. 空间注意力分散度 (Spatial Attention Dispersion)**

量化单步内 attention 的分布均匀程度，使用 entropy 度量：

```
H(A) = -Σᵢ A[i,:] · log(A[i,:])ᵀ
```

- **高 H（分散）**：attention 均匀分布在多个 token → 需要保留更多 KV 对
- **低 H（集中）**：attention 集中在少数 token → 可以大幅压缩

**2. 时间注意力偏移 (Temporal Attention Shift)**

量化 attention 热点在时间维度上的变化程度，使用 variance 度量：

```
V(A) = Σⱼ Var(A[:,j])
```

- **高 V（不稳定）**：token 重要性频繁变化，需保留更多候选 → 需要更大 cache
- **低 V（稳定）**：始终关注相同 token → 小 cache 即可

## 方法详解

### 1. Preference Score：统一度量 Layer 需求

将两个指标合并为单一的 **preference score** P，用于量化每层对 cache 资源的需求程度：

```
P = H^τ₁ · V^τ₂
```

其中 H 和 V 从 recent window（最近 Sᵥ 个 token 的 attention）计算：

```
H = H(A[-Sᵥ:, :-Sᵥ])
V = V(A[-Sᵥ:, :-Sᵥ])
```

温度参数 τ₁、τ₂ 调节两个因素的相对权重，适应不同模型和预算约束。

### 2. 自适应 Budget 分配

根据 preference score 按比例分配全局 budget：

```
Bₗ = (Pₗ / Σₖ Pₖ) · B_total
```

其中 Pₗ 为 layer l 的 preference score，B_total 为全局总预算。这确保 dispersion 高或 shift 大的 layer 自动获得更大份额。

### 3. Cascading Cache Management

![[images/CAKE_fig4_cascading_management.png]]

直接计算所有层的最优分配需要一次性 materialize O(S·L) 的 KV cache，可能超出内存预算。CAKE 采用 **级联式** 管理：

**Algorithm 1 核心流程：**

```
for stage m = 0 to L-1:
    1. 加入第 m 层 KV cache（full cache for current layer）
    2. 计算 layer m 的 preference score Pₘ
    3. 基于已有的 {P₀,...,Pₘ} 重新分配 budget B⁽ᵐ⁾ = {B₀⁽ᵐ⁾,...,Bₘ⁽ᵐ⁾}
    4. 对所有 layer 0..m 执行 eviction，按新 budget 裁剪
```

**关键性质**（Proposition 1）：每层 cache 的分配 budget 随 stage 推进**单调递减** → KV cache 始终是前一 stage 的子集 → eviction 只需从已保留的 token 中继续筛选，不需要回溯。

### 4. Attention-Shift Tolerant Eviction Indicator

对 layer l 中的 token n，定义 eviction importance：

```
I[n] = Mean(A[-Sᵥ:, n]) + γ · Var(A[-Sᵥ:, n])
```

- **Mean**：捕获持续重要性（长期被关注的 token）
- **γ · Var**：捕获注意力波动性（曾经或即将被关注的 token）
- **Recent window token**：最近 Sᵥ 个 token 赋予无穷大值 Ω，强制保留

Eviction 操作保留 top-Bₗ 个 importance 最高的 token：

```
D_l = TopK(I_l, B_l)
K̂_l = K_l[D_l, :],  V̂_l = V_l[D_l, :]
```

Ablation 研究表明 Mean+γ·Var（加法组合）优于单独 Mean、单独 Var 或 Mean·Var（乘法组合）。

## 实验结果

### LongBench 性能

![[images/CAKE_fig5_longbench_curves.png]]

在 LongBench 16 个数据集上（6 个任务类别），5 个模型、6 种 budget 设置下测试：

| Model | Budget | CAKE | SnapKV | PyramidKV | H2O | TOVA |
|-------|--------|------|--------|-----------|-----|------|
| Llama2-7B-Chat | 128L | **29.29** | 28.09 | 28.20 | 26.06 | 25.31 |
| Llama2-7B-Chat | 1024L | **32.98** | 32.50 | 32.55 | 31.05 | 31.46 |
| Llama3.1-8B-Instruct | 128L | **43.39** | 42.53 | 41.94 | 41.67 | 42.47 |
| Mistral-7B-Instruct-v0.3 | 128L | Best | - | - | - | - |

CAKE 在所有 budget 设置下均取得最优或次优结果，低 budget（64L-256L）下优势尤为显著。

### NeedleBench 32K

在 32K context 的检索和推理任务上：

- 仅用 **3.2% cache**（256L/32K·L）即保持模型 retrieval 能力
- Single-Needle Retrieval 甚至**超越 full cache**（说明适当剪枝可去除干扰信息）
- Multi-Needle Retrieval 优势最为明显（受益于 CAKE 平衡长期/短期信息的设计）

### Memory & Throughput

![[images/CAKE_fig7_memory_latency.png]]

在 Mistral-7B + FlashAttention-2 + A100 80GB 上测试（B_total=1024L）：

- **峰值内存**：128K context 下降低 ~48.63%（与 SnapKV/PyramidKV 相当，均为固定大小 cache）
- **Decoding 延迟**：128K 下实现约 **10× 加速**
- Full cache 在 256K 时 OOM，CAKE 等 eviction 方法可正常运行

### Ablation: Allocation Strategy

| 策略 | LongBench Avg. |
|------|---------------|
| Uniform | 28.36 |
| Pyramid | 28.69 |
| Random | 28.3±0.2 |
| **P2A (CAKE)** | **29.29** |

P2A 一致优于所有静态分配策略，包括 PyramidKV 的固定 pyramid 形状。

### Ablation: Eviction Indicator

| Indicator | Avg. |
|-----------|------|
| Mean only | 28.82 |
| Var only | 28.68 |
| Mean · Var | 28.60 |
| **Mean + γ·Var** | **29.29** |

加法组合（Mean + γ·Var）优于乘法组合和单一指标，因为加法允许两个信号独立贡献。

## 与其他方法的对比定位

| 维度 | CAKE | SnapKV | PyramidKV | H2O | KeyDiff |
|------|------|--------|-----------|-----|---------|
| **Cache 分配** | Layer-adaptive (P2A) | Uniform | Fixed pyramid | Uniform | Uniform |
| **Eviction 依据** | Mean+Var attention | Recent attention clustering | Same as SnapKV | Cumulative attention | Key cosine diversity |
| **需要 attention matrix** | 是（recent window） | 是 | 是 | 是 | **否** |
| **Prefill 管理** | Cascading（逐 stage） | 一次性 evict | 一次性 evict | 逐 token | Block processing |
| **Layer 异质性** | 显式建模 | 未考虑 | 固定规则 | 未考虑 | 未考虑 |
| **适用模型** | MHA + GQA | MHA + GQA | MHA + GQA | MHA | MHA + GQA |

CAKE 的核心差异化在于 **layer-level 自适应**——它是第一个系统性地建模 layer 间注意力异质性并据此动态分配 cache 预算的方法。这使得其在相同全局 budget 下能更高效地利用内存。

## 关键公式总结

```
# Spatial Dispersion (entropy)
H(A) = -Σᵢ A[i,:] · log(A[i,:])ᵀ

# Temporal Shift (variance)  
V(A) = Σⱼ Var(A[:,j])

# Preference Score
P = H^τ₁ · V^τ₂

# Budget Allocation
Bₗ = (Pₗ / Σₖ Pₖ) · B_total

# Eviction Indicator
I[n] = Mean(A[-Sᵥ:, n]) + γ · Var(A[-Sᵥ:, n])

# Eviction
D_l = TopK(I_l, B_l)
```

## 相关链接

- 代码：https://github.com/antgroup/cakekv
- 相关方法：[[Token Eviction]]、[[SnapKV]]、[[PyramidKV]]、[[H2O]]、[[KeyDiff]]、[[Expected Attention]]
- 综述：[[overview]]
