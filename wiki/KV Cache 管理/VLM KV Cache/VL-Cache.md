---
title: "VL-Cache: Sparsity and Modality-Aware KV Cache Compression for Vision-Language Model Inference Acceleration"
aliases: [VL-Cache, VLM KV Cache, Modality-Aware Cache Compression]
source_type: paper
source_url: "https://arxiv.org/abs/2410.23317"
source_date: 2024-10-29
ingested: 2026-05-14
venue: arXiv 2024
authors: [Dezhan Tu, Danylo Vashchilenko, Yuzhe Lu, Panpan Xu]
affiliation: UCLA / Amazon AWS AI
code: null
tags: [model-compression, kv-cache, eviction, VLM, modality-aware, sparsity, vision-language, layer-adaptive, token-scoring, inference-acceleration]
---

## 核心要点

VL-Cache 是首篇专门针对 **Vision-Language Model (VLM)** 优化 KV cache compression 的工作。核心发现：VLM 的 attention sparsity pattern 与纯 LLM **根本不同**——存在明显的 **modality boundary**（视觉 token 与语言 token 之间的注意力分界），直接迁移 LLM 的 eviction 方法会产生次优结果。

![[images/VL-Cache_fig1_attention_comparison.png]]

关键创新：

1. **VLM Attention Sparsity Profile**：发现 VLM 中视觉 token 之间注意力近似均匀，而语言 token 对视觉 token 高度集中在少数关键 token 上——存在清晰的 modality boundary
2. **Layer-Adaptive Sparsity-Aware Cache Budget Allocation**：利用 prefill 阶段的 post-vision attention 稀疏度动态分配每层 cache budget（非固定/非单调递减）
3. **Modality-Aware Token Scoring Policy**：用 Accumulated Post-vision Attention（而非 full accumulated attention）评估 token 重要性，避免视觉 token 的均匀注意力"淹没"真正关键 token 的信号

结果：仅 **10% KV cache** 即可达到接近 full cache 的精度（98%+）；端到端延迟加速 **2.33×**；decoding 加速 **7.08×**；KV cache GPU 内存降低 **90%**。

## 动机：VLM vs LLM 的 Attention 差异

### Modality Boundary 观察

对比 Figure 1 中 LLM（纯语言）和 VLM（视觉+语言）的 attention score matrix：

- **LLM**（左图）：attention 在 prefill 和 decoding 中表现类似——critical tokens 在 decoding 阶段主要与 prefill 中相同 token 对齐
- **VLM**（右图）：沿 query 维度出现清晰的 **modality boundary**——post-vision 语言 token 对视觉 token 的注意力模式，与 decoding token 对视觉 token 的注意力模式高度一致

**关键洞察**：在 VLM 中，language tokens（特别是 post-vision prompt tokens）的 attention pattern 远比 visual tokens 自身的 attention pattern 更能预测 decoding 时哪些 token 重要。

### 为什么 LLM 方法直接迁移到 VLM 效果差？

1. **Accumulated Attention 的 length bias**：视觉 token 之间的注意力近乎均匀分布 → 累积后所有视觉 token 分数差异很小 → 无法有效区分关键 vs 冗余视觉 token
2. **固定 budget 分配不适配**：VLM 不同层的 sparsity 变化为 70%–99%，远比 LLM 更异质化
3. **固定 sliding window 不合适**：post-vision 语言 prompt 长度随任务变化很大，固定 window size 无法适配

## 预备实验：Sparsity 分析

![[images/VL-Cache_fig2_3_sparsity.png]]

### Layer-Wise Sparsity（Figure 2）

在 LLaVA-Mistral-7B 上，使用 ThresholdFilter (p=1%) 测量每层的 attention sparsity：

- **Prefill**（图 2a）：前两层显著低稀疏度（高密度），中间某些层也有局部高密度峰，整体 **非单调**
- **Decoding**（图 2b）：与 prefill 趋势高度相似（Pearson ρ = 0.695），除第二层在 decoding 时变得更稀疏
- **非单调分布**：否定了 PyramidKV 的单调递减假设——某些中间层需要更多 cache budget

**结论**：Prefill 阶段的 sparsity 可以有效预测 decoding 阶段各层所需的 KV cache 大小。

### Cache Hit Rate（Figure 3）

定义 Cache Hit Rate = 在 eviction 后保留的 top-k decoding attention tokens 的召回率。比较三种 scoring policy：

| Scoring Policy | 策略 | Cache Hit Rate |
|------|------|------|
| Accumulated Attention (H2O) | 全 prompt 累积 attention | 中等，受 length bias 影响 |
| Accumulated Sliding Window | 最近 window 内累积 | 较好，但 window size 固定 |
| **Accumulated Post-vision Attention (ours)** | 仅 post-vision 语言 tokens 累积 | **最高**，跨所有层一致最优 |

## 方法：VL-Cache

![[images/VL-Cache_fig4_overview.png]]

### Step 1：Sparsity-Aware Cache Budget Allocation（Algorithm 1）

在 prefill 完成后，基于 post-vision attention 的 layer-wise sparsity 动态分配每层的 KV cache budget：

```
对每层 l:
  1. 计算 post-vision attention A' = softmax(Q_{m-τ:m} · K^T / √d)
  2. 应用 ThresholdFilter(A', p=1%) 计算 sparsity γ'(l)
  3. 按 (1 - γ'(l)) 的比例分配 cache budget: β(l) = clip((1-γ'(l))/(Z) × αL, 0.01, 1)
```

其中 α 是全模型的目标 cache 比例（如 10%），τ 是 post-vision 语言 prompt 长度。

**与 PyramidKV 的关键区别**：
- PyramidKV：所有 prompt 使用固定的单调递减 budget（prompt-independent）
- VL-Cache：每个 prompt 基于其实际 sparsity pattern 动态分配（prompt-dependent + non-monotonic）

### Step 2：Modality-Aware Token Scoring

给定每层的 budget k(l)，使用 Accumulated Post-vision Attention 对所有 cache tokens 打分，保留 top-k：

```
score_j = Σ_i A'_{ij}   (对 post-vision 语言 query i 求和)
```

**为什么 Post-vision Attention 更优？**
- 计算复杂度更低：O(τm) vs O(m²)，因为视觉 token 主导 prompt 长度（τ ≪ m）
- 更高的 Cache Hit Rate：post-vision 语言 token 的 attention 与 decoding token 的 attention 高度对齐
- 天然是 dynamic window：window size = post-vision 语言 prompt 长度（自适应 prompt 内容）

### 实现细节

- 使用 Triton kernel 实现 self-attention forward、layer-wise sparsity evaluation、modality-aware token scoring
- Budget allocation 仅在 prefill 后执行一次，开销可跨多步 decoding 摊销
- 保留 recent tokens（占 budget 的 10%）+ top-k scoring tokens

## 实验结果

### 精度评估（Figure 5 + Table 2）

![[images/VL-Cache_fig5_accuracy.png]]

在 3 个数据集 × 2 个模型上评估（LLaVA-Mistral-7B + LLaVA-1.6-34B）：

**Coco-Caption (CIDEr)**：
| Budget | VL-Cache | H2O | PyramidKV | StreamingLLM |
|--------|----------|-----|-----------|--------------|
| 5% | 82.53 | 64.45 | 23.21 | 11.82 |
| 10% | 100.36 | 90.36 | 66.41 | 33.98 |
| 20% | 102.06 | 104.04 | 80.76 | 91.87 |
| 100% | 100.68 | 100.68 | 100.68 | 100.68 |

**DocVQA (ANLS)**：10% budget 下 VL-Cache 62 vs H2O 56 vs PyramidKV 60 vs StreamingLLM 47

**MathVista (ACC)**：10% budget 下 VL-Cache 39 vs H2O 36 vs PyramidKV 38 vs StreamingLLM 33

关键结论：VL-Cache 在 **5-10% 极低 budget** 下远超所有 baseline，是唯一在 10% cache 即达到接近 full-cache 精度的方法。

### 速度评估（Table 3 + Figure 6）

![[images/VL-Cache_table3_fig6_speed.png]]

10% KV cache budget 下的性能（LLaVA-Mistral-7B，100 output tokens）：

| Batch Size | Prompt Length | Prefill Speedup | Decoding Speedup | End-to-End Speedup |
|------------|-------------|-----------------|------------------|--------------------|
| 1 | 2K | 0.96× | 1.19× | 1.16× |
| 1 | 32K | 0.99× | 3.32× | 1.85× |
| 1 | 128K | 0.99× | **7.08×** | 1.66× |
| 4 | 32K | 0.99× | 6.07× | 2.06× |
| 64 | 2K | 0.98× | 5.23× | **2.33×** |

特点：
- Prefill 开销极低（1-6%）：因为 post-vision attention 计算是 O(τm) << O(m²)
- Decoding 加速随 context length 和 batch size 线性增长
- End-to-end 加速受 prefill 延迟 bound（因 prefill 未压缩）
- 长输出任务（caption、description、CoT reasoning）收益最大

### 并发能力

KV cache 压缩 90% 后，理论上可支持 **10× 更高并发度**（在 KV cache 为显存瓶颈的场景下）。Figure 6 展示在相同延迟目标下，VL-Cache 提供更高的 server-level throughput。

## 与相关方法的关系

- **[[H2O]]**：VL-Cache 的直接改进对象——H2O 的 Accumulated Attention scoring 在 VLM 中表现次优（受 visual token length bias 影响）
- **[[PyramidKV]]**：VL-Cache 的 budget allocation 直接改进 PyramidKV 的固定递减策略——变为 prompt-dependent + non-monotonic
- **[[StreamingLLM]]**：Sink + Recent 策略在 VLM 低 budget 下严重退化——无法捕捉关键视觉 token
- **[[Token Eviction]]**：VL-Cache 属于 Token Eviction 的 VLM 特化版本，核心创新在于 modality-aware scoring
- **[[DuoAttention]]**：DuoAttention 做 head-level 分类（retrieval vs streaming），VL-Cache 做 modality-level 分类（vision vs language scoring）；两者可能互补
- **[[KV Cache Quantization]]**：与 VL-Cache 完全正交——量化降低 per-token bits，VL-Cache 减少 token 数量，可组合
- **[[Q-VLM]]**：Q-VLM 做 VLM 的 weight quantization（跨层依赖分析），VL-Cache 做 VLM 的 KV cache eviction——两者从不同角度优化 VLM 推理
- **[[VLM Quantization Best Practices]]**：该工作发现 VLM 各组件量化的非加性交互效应，与 VL-Cache 发现的 modality boundary 现象互相呼应

## 关键发现对 VLM KV Cache 研究的启示

1. **Modality boundary 是 VLM 特有的结构**：vision tokens 之间均匀 attend，language tokens 对 vision tokens 高度集中——这意味着 VLM 需要 modality-specific 的压缩策略
2. **Post-vision attention 是更好的 proxy**：比 full accumulated attention 和 sliding window 都更能预测 decoding 时的 token 重要性
3. **Layer-wise sparsity 非单调**：否定了"浅层需要更多 cache"的简单假设；需要 per-prompt 动态分配
4. **VLM 极端压缩可行**：10% cache 即可保持 98%+ 精度——因为视觉 token 中真正重要的极少（大部分是冗余的 patch tokens）
5. **对 VLM 研究方向的暗示**：modality-aware 差异化策略（而非统一的 token-level eviction）是 VLM 推理优化的关键方向

## 局限

- 仅在 LLaVA 系列评估，未涵盖最新的 VLMs（如 Qwen-VL、InternVL 等）
- 未探索与量化的组合（仅做 eviction）
- Prefill 阶段不加速（仅加速 decoding）——对 prefill-bound 场景收益有限
- Budget allocation 需要一次性计算全部 post-vision attention，对极长 prompt 可能有内存压力
- 未讨论 video VLM 中跨帧的 temporal redundancy 利用
