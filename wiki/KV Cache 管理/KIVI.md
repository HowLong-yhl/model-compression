---
title: "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
aliases: [KIVI, Liu 2024 KIVI]
source_type: paper
source_url: "https://arxiv.org/abs/2402.02750"
source_date: 2024-02-05
ingested: 2026-04-28
venue: ICML 2024
authors: [Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu]
affiliation: Rice University, HKUST, Johns Hopkins University, CMU, Meta
code: "https://github.com/jy-yuan/KIVI"
tags: [model-compression, quantization, method, kv-cache, tuning-free, asymmetric, 2-bit, inference]
---

## 核心要点

KIVI 是一个 **tuning-free** 的 KV cache 2-bit 量化方案，核心发现是 Key 和 Value 应该沿**不同方向**量化——Key per-channel，Value per-token。这种非对称设计在 2-bit 下即可近无损运行，实现 **2.6× 内存压缩**和最高 **4× batch size** 提升。

核心洞察来自对 KV cache 数值分布的精细分析：Key cache 的 outlier 集中在固定通道（channel）上，per-channel 量化方向可以让每个通道有独立的 scale/zero-point 来适配其独特分布；Value cache 没有固定 outlier 通道，但 attention 的稀疏性（softmax 权重集中在少数 token 上）使得 per-token 方向的量化误差只在低权重 token 上累积，不影响最终输出。

## 方法详解

### Key 分析：固定 Outlier Channel

通过可视化 LLaMA-2-7B 各层 Key cache 的 channel 分布，发现：

1. **跨 token 一致性**：同一通道在不同 token 上的值域高度稳定（标准差小）
2. **跨通道差异性**：不同通道间的值域差异极大（某些通道集中在 [-0.5, 0.5]，其他在 [-10, 10]）
3. **与 LLM 权重 outlier 的一致性**：outlier channel 的位置与 [[Emergent Outlier Features]] 报告的位置一致

**结论**：Key 应该 per-channel 量化——每个通道有独立的 (scale, zero-point) 对。

### Value 分析：无固定 Outlier + Attention 稀疏性

Value cache 的分布特性与 Key 完全不同：

1. **无固定 outlier 通道**：Value 的通道分布随 token 变化，没有稳定的 outlier 模式
2. **Per-token 量化误差分析**：per-token 量化的误差 $E_v$ 对最终输出 $Y = \text{softmax}(QK^T) \cdot V$ 的影响为 $\Delta Y = A \cdot E_v$，其中 A 是 attention 权重矩阵
3. **Attention 稀疏性保护**：由于 softmax 输出高度稀疏（少数 token 占主导），per-token 量化误差在低权重 token 上的累积不会显著影响输出

**形式化**：对于 per-token 量化，第 i 个 token 的量化误差为 $e_i$，输出误差为 $\Delta y = \sum_i A_{:,i} \cdot e_i$。当 $A_{:,i} \approx 0$（低权重 token），即使 $e_i$ 较大也对输出无影响。

### Grouped Quantization + Residual

为进一步提升精度，KIVI 采用 **group quantization + sliding window residual** 的两级结构：

1. **Group quantization**：将量化维度分成大小为 G 的 group，每个 group 有独立的 (scale, zero-point)
2. **Residual（全精度缓冲区）**：保留最近 R 个 token 的 KV cache 为全精度，仅对旧 token 做量化
3. **量化时机**：当全精度缓冲区积累到 R 个 token 时，对最早的一批 token 做量化并合并到低精度区

**默认配置**：2-bit，Key group size = 32（per-channel），Value group size = 32（per-token），residual size = 128

### Tiled Matrix Multiplication

为支持混合精度矩阵乘法（低精度 KV × 全精度 Q），KIVI 实现了 tiled matmul kernel：

```
Output = [Attention(Q, K_quant, V_quant)] + [Attention(Q, K_fp, V_fp)]
                 低精度部分                         全精度 residual 部分
```

两部分结果通过 online softmax（类似 FlashAttention）合并。

## 实验结果

### WikiText-2 Perplexity

| 模型 | FP16 | KIVI 2-bit | 退化 |
|------|------|-----------|------|
| LLaMA-2-7B | 5.47 | 5.56 | +0.09 |
| LLaMA-2-13B | 4.88 | 4.95 | +0.07 |
| LLaMA-2-70B | 3.32 | 3.34 | +0.02 |
| Mistral-7B | 5.25 | 5.33 | +0.08 |

2-bit 下退化极小（<0.1 PPL），大模型退化更小。

### 长序列任务：Needle-in-a-Haystack

在 128K context length 的 Needle-in-a-Haystack 测试中，KIVI 2-bit 保持了与 FP16 几乎相同的检索准确率——证明量化没有损害模型的长程记忆能力。

### 系统性能

| 指标 | FP16 | KIVI 2-bit | 提升 |
|------|------|-----------|------|
| Peak Memory (LLaMA-2-7B, BS=1) | 100% | **38.5%** | 2.6× 压缩 |
| Max Batch Size (A100, LLaMA-2-7B) | BS=N | **BS=4N** | 4× |
| Throughput (LLaMA-2-7B, 4K ctx) | 1× | **2.35×** | — |
| Throughput (LLaMA-2-7B, 128K ctx) | 1× | **3.47×** | — |

长上下文场景下吞吐量提升更大（3.47× vs 2.35×），因为 KV cache 占总内存的比例更高。

### 消融实验

**量化方向的影响**（2-bit，LLaMA-2-7B）：

| Key 方向 | Value 方向 | PPL |
|---------|-----------|-----|
| Per-channel | Per-token | **5.56** |
| Per-token | Per-channel | 8.14 |
| Per-token | Per-token | 7.91 |
| Per-channel | Per-channel | 6.03 |

最优组合（per-channel Key + per-token Value）比最差组合（per-token Key + per-channel Value）低 **2.58 PPL**，验证了非对称量化方向的必要性。

**Group size 和 Residual size 的影响**：

| Group size | Residual size | 2-bit PPL |
|-----------|--------------|-----------|
| 32 | 128 | **5.56** |
| 64 | 128 | 5.71 |
| 128 | 128 | 6.03 |
| 32 | 64 | 5.62 |
| 32 | 0 | 5.93 |

Group size = 32 和 Residual size = 128 是精度-效率的最优平衡点。

## 局限性

- **Residual buffer 增加存储**：128 个 FP16 token 的存储量等于约 1024 个 2-bit token，在极端长上下文下 overhead 比例很小但非零
- **Per-channel Key 需要完整 token 序列**：per-channel 量化需要在通道维度上积累足够 token 才能估计 scale/zero-point，初始阶段统计不稳定
- **未处理 RoPE 影响**：与 [[KVQuant]] 不同，KIVI 未采用 Pre-RoPE 策略（per-channel 方向部分缓解了 RoPE 影响，但不完美）
- **仅支持 asymmetric uniform quantization**：未探索 non-uniform quantization（如 k-means），可能在 2-bit 下有进一步收益
- **Tiled matmul 的效率**：混合精度矩阵乘法比纯 FP16 或纯 INT 的 matmul 效率低，custom kernel 有优化空间

## 与已有知识的关联

### 与 KVQuant 的互补

KIVI 和 [[KVQuant]] 独立发现了相同的核心 insight——Key 应 per-channel、Value 应 per-token 量化。两篇论文几乎同时发表（2024 年 1-2 月），验证了这一发现的鲁棒性。差异在于：

- KVQuant 追求极致精度（NUQ + sensitivity weighting），面向超长上下文（10M tokens）
- KIVI 追求极致简洁（tuning-free + uniform 量化），面向通用推理加速

### 非对称量化方向的理论基础

KIVI 的 per-channel Key / per-token Value 发现可以从 [[Emergent Outlier Features]] 的角度理解——LLM 的 outlier 主要沿特定通道出现（由 LayerNorm 和 Residual Stream 的交互导致），Key 作为 QKV 投影的输出继承了这种 channel-wise 结构；而 Value 经过 attention-weighted 聚合后分布更均匀。

### Attention 稀疏性的利用

KIVI 对 Value per-token 误差的分析利用了 attention 的稀疏性——这与 H2O（Heavy Hitter Oracle）和 SnapKV 等 KV cache eviction 方法的理论基础一致。eviction 方法直接丢弃低权重 token 的 KV，KIVI 则保留所有 token 但以低精度存储，两者利用了同一个观察。

## 引用关系

← 理论基础：[[Emergent Outlier Features]]（Key 的固定 outlier channel），Attention Sparsity（Value per-token 量化的理论支撑）
← 量化技术：Asymmetric Uniform Quantization，Group Quantization（[[Quantization Granularity]]）
← 相关方法：[[KVQuant]]（同样发现 per-channel Key 有效），GEAR（低秩 + 稀疏误差补偿）
→ 贡献：首个 tuning-free 2-bit KV cache 量化方案，明确了 Key/Value 的非对称量化方向
→ 后续影响：RotateKV（IJCAI 2025）和 KVLinC（2025）在 KIVI 基础上引入旋转进一步提升 2-bit 精度
→ 概念关联：[[KV Cache Quantization]]（本方法是 KV cache 量化的代表作）
