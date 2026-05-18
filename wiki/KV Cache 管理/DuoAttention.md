---
title: "DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads"
aliases: [DuoAttention, Retrieval Heads, Streaming Heads, Duo Attention]
source_type: paper
source_url: "https://arxiv.org/abs/2410.10819"
source_date: 2024-10-14
ingested: 2026-05-13
venue: ICML 2025
authors: [Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Junxian Guo, Shang Yang, Haotian Tang, Yao Fu, Song Han]
affiliation: MIT / Tsinghua University / SJTU / University of Edinburgh / NVIDIA
code: "https://github.com/mit-han-lab/duo-attention"
tags: [model-compression, kv-cache, eviction, attention-heads, head-classification, retrieval-heads, streaming-heads, long-context, optimization-based, system]
---

## 核心要点

DuoAttention 基于一个关键观察：LLM 的 attention heads 可以被分为两类——**Retrieval Heads**（需要完整 KV cache 来检索长距离上下文信息）和 **Streaming Heads**（仅依赖 attention sinks + recent tokens）。只对 retrieval heads 保留完整 KV cache，streaming heads 使用恒定长度的轻量 cache，即可在几乎不损失长上下文能力的前提下大幅降低内存和延迟。

![[images/DuoAttention_fig1_attention_maps.png]]

关键结论：

1. **Head 二分类**：MHA 模型（Llama-2-7B）仅 ~25% heads 是 retrieval heads；GQA 模型（Llama-3-8B）~50% KV heads 是 retrieval heads
2. **修改 streaming heads 影响极小**：将 streaming heads 限制为 sink+recent 后，passkey 检索准确率几乎不变；反之修改 retrieval heads 则性能崩溃
3. **Optimization-based 识别**：比 attention profiling 更准确——直接优化 gate value 测量 end-to-end 输出偏差，而非仅看 attention score
4. **高效部署**：Chunked pre-filling 下 streaming heads 实现 O(LK) 时间 + O(K) 内存（K 为 chunk size）

结果：MHA 25% retrieval ratio → 2.55× 内存压缩 + 2.18× 解码加速；GQA 50% → 1.67× 内存 + 1.50× 解码加速。结合 8-bit 权重 + 4-bit KV 量化，Llama-3-8B 可在**单卡 A100 处理 3.3M tokens**。

## 动机：Retrieval vs Streaming Heads

### 两类 Attention Head 的观察

![[images/DuoAttention_fig4_gate_values.png]]

在 Transformer-based LLMs 中，attention heads 展现出截然不同的行为模式：

- **Retrieval Heads**：在解码时捕获上下文中语义相关的 token（如 "best fruit" → "orange"）。这些 head 执行长距离信息检索，压缩它们的 KV cache 会导致关键上下文信息丢失。
- **Streaming Heads**：主要关注 initial tokens（attention sinks）和 recent tokens，不强调远距离上下文相关性。这些 head 的行为类似 StreamingLLM 的 sink+recent 模式。

关键发现（Figure 4 gate values）：
- Llama-2-7B（MHA, 32 heads/layer）：大部分 heads 的 gate value 很低（蓝色 = streaming），仅少量高值（红色 = retrieval）→ retrieval ratio ~25%
- Mistral-7B / Llama-3-8B（GQA, 8 KV heads/layer）：较多 heads 为 retrieval → retrieval ratio ~50%
- Llama-3-70B（GQA, 8 KV heads/layer, 80 layers）：类似的 ~50% pattern

**MHA vs GQA 的差异解释**：GQA 中每个 KV head 服务多个 query heads，因此单个 KV head 承担更多功能，更可能包含 retrieval 功能 → GQA 模型的 retrieval ratio 更高。

## 方法：Optimization-based Head Identification

![[images/DuoAttention_fig2_overview.png]]

### Phase 1：训练 Gate Values

为每个 KV head (i, j) 分配可训练的 gate value α_{i,j} ∈ [0, 1]：

```
output = α · FullAttention(Q, K, V) + (1-α) · StreamingAttention(Q, K, V)
```

其中 StreamingAttention 仅使用 sink tokens + recent tokens 的 KV。

**训练目标**：最小化与 full attention 模型输出的偏差 + L1 正则化鼓励稀疏（更多 streaming heads）：

```
L = L_distill + λ · L_reg

L_distill = (1/N) Σ Σ (H_full[j] - H_mixed[j])²    # 仅在 passkey tokens 上计算
L_reg = ΣΣ |α_{i,j}|                                 # L1 鼓励低 gate value
```

λ = 0.05。

### 合成数据设计

不同于使用自然语言建模，DuoAttention 设计了专门的**合成 passkey 检索数据集**：

- 在很长的文本中嵌入 10 个随机生成的 32-word passkeys
- 模型需要在最后回忆所有 passkeys
- Distillation loss 仅在 passkey tokens 上计算

优势：每个监督信号都直接相关于长上下文检索能力，比自然语言建模更高效（ablation 验证）。

### 训练效率

- 可训练参数仅有 N_layers × H_heads 个浮点数（约数千个）
- 2000 steps 即可完成
- 8×A100 GPU 数小时内完成

### Phase 2：部署时二值化

根据阈值 τ 将 gate values 二值化：
- α > τ → Retrieval Head → Full KV cache
- α ≤ τ → Streaming Head → Sink + Recent tokens only

τ 通过 Pareto 最优分析确定（accuracy vs compression ratio）。

## 部署设计

![[images/DuoAttention_fig5_deployment.png]]

### Decoding

每层分配两个 KV cache：
- **Retrieval heads KV cache**：存储所有 past tokens（与标准 KV cache 相同）
- **Streaming heads KV cache**：仅存储 attention sinks + recent tokens（固定大小）

新 token 进入时，query/key/value 按 head 维度切分 → 分别计算 full attention 和 streaming attention → 拼接输出。

### Chunked Pre-filling

DuoAttention 完全兼容 chunked pre-filling。Streaming heads 的 pre-filling 在每个 chunk 计算后立即 prune，仅保留 sink + recent：

- 时间复杂度：从 O(L²) 优化为 **O(LK)**（K = chunk size）
- 内存复杂度：从 O(L) 优化为 **O(K)**（常数内存）
- 无需特殊 kernel，直接使用 FlashAttention-2

### Head Reordering 优化

部署前将 Q/K/V projection weights 的输出通道重新排列——retrieval heads 和 streaming heads 分别聚为连续组。这样 KV cache 操作变为高效的 slicing/concatenation（而非 scatter/gather）。

## 实验结果

### Needle-in-a-Haystack

![[images/DuoAttention_fig6_niah.png]]

- Llama-2-7B（MHA, 25% retrieval）：在 32K context 下与 Full Attention 完全一致，H2O/StreamingLLM/TOVA 在 25% budget 下均有明显空洞
- Llama-3-8B（GQA, 50% retrieval）：在 1048K context 下与 Full Attention 一致，其他方法均有大面积检索失败

### LongBench

在 14 个 LongBench 任务上（Figure 7），DuoAttention 展现优越的 KV budget-accuracy trade-off：
- Llama-2-7B 25% budget 在大多数任务上接近 full attention
- Llama-3-8B 50% budget 在几乎所有任务上等价于 full attention
- 显著优于 H2O、StreamingLLM、TOVA 在相同 budget 下的表现

### Short Benchmarks

| Model | Budget | MMLU | MBPP | MT-Bench |
|-------|--------|------|------|----------|
| Llama-3-70B Full | 100% | 79.38% | 47.85% | 8.93 |
| H2O | 50% | 79.26% | 32.12% | 7.16 |
| TOVA | 50% | 79.15% | 36.09% | 7.96 |
| StreamingLLM | 50% | 77.46% | 5.57% | 5.41 |
| **DuoAttention** | **50%** | **79.35%** | **47.09%** | **9.14** |

DuoAttention 在 short benchmarks 上也几乎无损，MBPP（代码生成）甚至优于 H2O/TOVA 30+ 个百分点。

### 效率结果

![[images/DuoAttention_fig9_decoding_efficiency.png]]

**Decoding**（单卡 A100）：
| 模型 | Retrieval Ratio | 内存压缩 | 延迟加速 |
|------|----------------|---------|---------|
| Llama-2-7B (MHA) | 25% | 2.55× | 2.18× |
| Llama-3-8B (GQA) | 50% | 1.67× | 1.50× |

**Pre-filling**（chunked）：
| 模型 | Retrieval Ratio | 延迟加速 | 内存压缩 |
|------|----------------|---------|---------|
| Llama-2-7B (MHA) | 25% | 1.73× | 2.38× |
| Llama-3-8B (GQA) | 50% | 1.63× | 1.53× |

**极端长上下文**：结合 8-bit weight quantization + 4-bit KV cache quantization + DuoAttention，Llama-3-8B 可在单卡 A100-80G 上解码 **3.3M tokens**（vs full attention FP16 仅 0.52M）。

## Ablation Studies

### Identification 方法对比

| 方法 | 效果 |
|------|------|
| Attention profiling（看 attention score） | 次优——忽略 value 的作用和 end-to-end 影响 |
| Language modeling loss | 自然文本中 long-range 信号稀疏，效率低 |
| **Optimization w/ synthetic data (ours)** | 最优——每个监督信号都直接相关 |

### Sink/Recent tokens 数量

- 训练时：Sink=128, Recent=256 → 充分识别
- 部署时：Sink=16, Recent=64 即可达到接近最优效果；更多带来边际收益

### 为什么 attention profiling 不够？

注意力分数不能完全反映 head 的实际重要性：
1. 忽略了 Value states 的角色——attention 大不代表 value 对 output 贡献大
2. 忽略了 end-to-end 影响——某些 head 的 attention 分布变化后对最终输出影响很小
3. 忽略了层间变异性——不同层的 attention 尺度不同，阈值难以统一

## 与相关方法的关系

- **[[StreamingLLM]]**：DuoAttention 的 streaming heads 本质上就是 StreamingLLM 的 sink+recent 策略，但 DuoAttention 只对 streaming heads 应用（而非所有 heads），保留了 retrieval 能力
- **[[H2O]] / [[TOVA]]**：基于 attention score 的 token-level eviction；DuoAttention 在 head-level 做二分类，更粗粒度但更稳定（不会误删 retrieval head 的关键 token）
- **[[Token Eviction]]**：DuoAttention 可视为 head-aware 的 eviction——不同 head 使用不同 eviction 策略（full vs streaming），而非统一策略
- **[[CAKE]]**：CAKE 也做 layer-adaptive budget 分配，但 CAKE 是按层分配 total budget，DuoAttention 是按 head 二分类
- **[[FastGen]]**：也做 per-head attention profiling 来 discard tokens，但基于 attention score 而非 optimization-based identification
- **[[KV Cache Quantization]]**：与 DuoAttention 完全正交——DuoAttention 减少需要缓存的 token 数（对 streaming heads），量化降低每个 token 的比特数。两者组合实现 3.3M context
- **[[GQA]]**：DuoAttention 完全兼容 GQA，且发现 GQA 模型的 retrieval ratio 更高（~50% vs MHA ~25%）
- **[[PyramidKV]]**：层自适应 budget 分配；DuoAttention 的 head-level 分类可以视为更细粒度的版本

## 关键贡献与局限

**贡献：**
- 首次系统化地将 attention heads 分为 Retrieval 和 Streaming 两类，并量化验证其对长上下文的不同影响
- 提出 optimization-based 识别方法，直接优化 end-to-end output deviation（而非依赖 attention score proxy）
- 合成 passkey 数据集设计——每个监督信号都直接相关长上下文检索
- 部署设计兼容 FlashAttention、chunked pre-filling、GQA，无需特殊 kernel
- 与量化完全正交，组合后支持百万级 context

**局限：**
- GQA 模型压缩率受限（~50% retrieval ratio → 仅 1.67× 压缩）——因为 GQA 每个 KV head 服务更多 query heads
- Head 分类是静态的（one-shot identification）——不随输入动态调整
- 无法减少 retrieval heads 的计算量——这些 head 仍需完整 O(L) attention
- 不适用于所有 heads 都是 retrieval 的极端情况（虽然实际 LLMs 中未观察到）
