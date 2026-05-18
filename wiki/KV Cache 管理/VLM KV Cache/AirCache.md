---
title: "AirCache: Activating Inter-modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference"
aliases: [AirCache, Activating Inter-modal Relevancy]
source_type: paper
source_url: "https://arxiv.org/abs/2503.29356"
source_date: 2025-07-23
ingested: 2026-05-14
venue: arXiv 2025
authors: [Kai Huang, Hao Zou, Bochen Wang, Ye Xi, Zhen Xie, Hao Wang]
affiliation: Alibaba Group
code: null
tags: [model-compression, kv-cache, eviction, VLM, modality-aware, observation-window, layer-adaptive, budget-allocation, inference-acceleration]
---

## 核心要点

AirCache 是阿里巴巴提出的 VLM 专用 KV cache 压缩方法，核心创新在于通过 **跨模态关联性 (Inter-modal Relevancy)** 来指导视觉 token 的重要性评估。与直接使用全部 text tokens 或连续 local window 作为观测窗口不同，AirCache 通过 text tokens 间的 self-attention 筛选出 **关键文本 tokens（Elite Observation Window）**，用这些语义上最重要的 text tokens 来评估视觉 token 的重要性，从而获得更一致和稳定的评分。

![[images/AirCache_fig1_motivation.png]]

*Figure 1: 不同观测窗口的比较。(a) LLaVA-OV-7B 解码阶段 attention map；(b) 全部 text tokens 作为观测窗口——评分不一致；(c) 最后 16 个 text tokens——评分噪声大；(d) 最后 16 个 visual tokens——缺乏文本语义指导；(e) AirCache 的 elite observation window——一致性最强。*

## 方法

![[images/AirCache_fig2_overview.png]]

### 组件一：Elite Observation Window

核心思想：不同 text tokens 对 visual tokens 的"评价"差异极大（如一些 text tokens 关注背景，另一些关注前景），直接用全部或局部连续 text tokens 评估会引入大量噪声。AirCache 选择语义上最关键的 text tokens 作为评估者。

**选择过程**：

1. 计算 instruction text tokens 间的 self-attention 矩阵：$A_{tt} = \text{Softmax}(\frac{Q_t K_t^T}{\sqrt{D}})$

2. 以最后一个 text token 为参考，筛选高 attention score 的 key text tokens：$k = \{j \mid A_{tt}[N_t-1, j] \geq \alpha \cdot \max A_{tt}[N_t-1, :]\}$，其中 $\alpha = 0.9$

3. 用选出的 key text tokens 的 query 与 visual tokens + key text tokens 的 key 计算跨模态 attention：$A_{vtk} = \text{Softmax}(\frac{Q_{tk} K_{vtk}^T}{\sqrt{D}})$

4. 对 attention 矩阵沿 text 维度平均池化得到 visual token importance：$I_v = \frac{1}{N_{tk}} \sum_{j=0}^{N_{tk}-1} A_{vtk}[j, :N_v]$

**优势**：Elite observation window 中的 text tokens 倾向于对同一 visual token 给出更一致的评分，从而提升基于投票机制的 ranking 稳定性。

### 组件二：Layer-wise KV Cache Budget Allocation

![[images/AirCache_fig3_distribution.png]]

*Figure 3: (a) 各层视觉 token importance 总和（strength）差异显著；(b) 按重要性排序逐个保留 visual token 的性能——仅 ~10% 有显著正面影响；(c) 各层 importance 分布差异——head effect 程度不同。*

从两个角度量化各层的 budget 需求：

**Strength（强度）**——该层对视觉信息的关注程度：
$$s_t = \sum_{i=0}^{N_v-1} I_v[i]$$

**Skewness（偏度）**——该层 attention 分配的"头部效应"程度：
$$s_k = \frac{N_v}{(N_v-1)(N_v-2)} \sum_{i=1}^{N_v} \left(\frac{I_v[i] - \mu_{I_v}}{\sigma_{I_v}}\right)^3$$

最终各层 budget 通过归一化后组合：
$$\hat{r} = \frac{1}{2}(s_t' + s_k') \cdot r$$

其中 $s_t'$ 和 $s_k'$ 是跨层归一化后的 strength 和 skewness。直觉：对视觉信息关注度高（strength 大）且 attention 分配集中（skewness 大，head effect 明显）的层分配更多 budget。

### 执行流程

1. Prefill 完成后，所有层的完整 KV cache 暂时保留
2. 对所有层计算 importance scores、strength、skewness
3. 跨层归一化后确定每层 budget
4. 按 importance ranking 一次性压缩各层 visual KV cache
5. 仅压缩 visual tokens，text tokens 全部保留

## 实验结果

### 主要精度评估（Table 1）

![[images/AirCache_table1_main_results.png]]

在三款架构不同的 7-8B VLMs 上，四个 VQA 数据集的结果：

**LLaVA-OV-7B (10% visual cache)**：
- ChatQA: 79.9 (Full: 80.3)，-0.5%
- InfoVQA: 65.7 (Full: 66.1)，-0.6%
- DocVQA: 85.5 (Full: 87.0)，-1.7%
- TextVQA: 75.3 (Full: 76.0)，-0.9%

**关键对比（1% visual cache, LLaVA-OV-7B）**：
| Method | ChatQA | InfoVQA | DocVQA | TextVQA |
|--------|--------|---------|--------|---------|
| Full | 80.3 | 66.1 | 87.0 | 76.0 |
| H2O | 71.0 | 52.0 | 55.3 | 60.4 |
| SnapKV | 72.9 | 57.8 | 64.1 | 58.2 |
| **AirCache** | **76.4** | **62.5** | **73.2** | **67.1** |

AirCache 在 1% cache 下平均比 SnapKV 高 6.5%，优势在高压缩比下更加显著。

### 推理速度（Table 3）

![[images/AirCache_table3_speed.png]]

LLaVA-OV-7B，输出固定 512 tokens：

| Batch Size | Prompt | 50% Decode ↓ | 10% Decode ↓ | 10% Throughput ↑ |
|------------|--------|-------------|-------------|------------------|
| 8 | 2K | -19.0% | -29.3% | +41.6% |
| 8 | 16K | -30.9% | -47.5% | +108.5% |
| 8 | 32K | -38.7% | -65.7% | +192.1% |
| 16 | 8K | -31.0% | -47.8% | +91.7% |
| 16 | 16K | -37.7% | -65.3% | +188.3% |

Prefill 开销：约 4-18%（因需存完整 KV cache 后一次性压缩）。

### 与 Token Pruning 方法对比

| Method | ChatQA 10% | ChatQA 1% | Prefill | Decoding |
|--------|-----------|----------|---------|----------|
| Full | 80.3 | 80.3 | 9.4s | 22.6s |
| FastV | 47.7 | 16.9 | 5.4s | 11.7s |
| FasterVLM | 50.1 | 18.8 | 5.2s | 11.3s |
| IVTP | 55.8 | 22.5 | 5.8s | 12.6s |
| **AirCache** | **79.9** | **76.4** | 9.8s | 11.8s |

Token pruning 在 prefill 阶段删除 token 导致严重视觉信息丢失；AirCache 在完整 prefill 后仅压缩 KV cache，保留了跨模态交互信息。

### 模型规模验证（InternVL2 系列）

InternVL2-1B / 4B / 26B 均验证有效。规律：模型越小，KV cache 压缩对性能影响越大（小模型 token 间信息整合能力更弱，更依赖完整 cache）。

## 消融实验

### 观测窗口对比（1% cache, LLaVA-OV-7B）

| 观测窗口策略 | ChatQA | DocVQA |
|------------|--------|--------|
| 连续 16 text tokens | 70.4 | 61.3 |
| 连续 32 text tokens | 72.9 | 64.1 |
| 全部 text tokens | 72.2 | 65.7 |
| 连续 visual tokens (32) | 68.8 | 59.2 |
| **Elite observation window** | **76.4** | **73.2** |

Elite observation window 大幅优于其他策略，尤其比 visual-only window 高 7-14 分。

### Budget Allocation 策略对比

| 分配策略 | ChatQA | DocVQA |
|---------|--------|--------|
| Average (均匀) | 72.2 | 69.9 |
| Pyramid (递减) | 69.6 | 55.8 |
| Only strength | 74.2 | 71.1 |
| Only skewness | 74.7 | 71.9 |
| **Strength + Skewness** | **76.4** | **73.2** |

PyramidKV 的递减策略在 VLM 中表现最差——验证了 VLM 层间 sparsity 非单调的发现。

### Vision-only vs. Unified Compression

仅压缩 visual tokens 远优于统一压缩（visual + text）。统一压缩破坏 text instruction 的完整表示，导致模型产生错误输出和重复倾向，推理时间反而增加。

### Drop vs. Merge

在 LLaVA-OV / InternVL2 / Qwen2-VL 系列中，直接 drop 比 merge 效果更好。Merge 会在这些模型中导致重复输出问题（仅在较老的 LLaVA-v1.5 上 merge 有效）。

## 与相关方法的关系

- **[[VL-Cache]]**：同为 VLM 专用 KV cache eviction。VL-Cache 用 post-vision attention（视觉段之后的语言 token）评分，AirCache 用 elite observation window（self-attention 筛选的关键 text tokens）评分。AirCache 在模型多样性验证上更广泛（InternVL2、Qwen2-VL），VL-Cache 仅在 LLaVA 系列。两者的 budget allocation 思路不同：VL-Cache 基于 sparsity（1-γ'），AirCache 基于 strength+skewness
- **[[SnapKV]]**：LLM 的 observation-window voting 方法。AirCache 改进了 observation window 的选择方式——从"连续局部 text tokens"升级为"self-attention 筛选的关键 text tokens"
- **[[PyramidKV]]**：AirCache 明确验证了 Pyramid 递减策略在 VLM 中效果差（甚至不如均匀分配），进一步证明 VLM 需要 per-prompt adaptive budget
- **[[H2O]]**：Full-range accumulated attention 方法，在 VLM 中缺乏跨模态感知
- **[[DuoAttention]]**：Head-level 分类。AirCache 未做 head-level 区分，但两者可能互补
- **[[Token Eviction]]**：AirCache 属于 Token Eviction 的 VLM 特化，仅在 decode 阶段压缩 visual KV cache
- **Token Pruning (FastV, IVTP)**：在 prefill 前/中就删除 visual tokens，更激进但信息损失大。AirCache 保留完整 prefill 交互再压缩 KV，精度远优

## 关键发现

1. **Visual token 冗余极高**：约 90% 的 visual tokens 对模型输出无显著正面贡献（Figure 3b）
2. **Elite observation window 一致性最强**：关键 text tokens 对同一 visual token 的评分高度一致，voting-based ranking 噪声最低
3. **VLM 层间 budget 需求差异大**：Strength 和 skewness 在不同层差异显著，PyramidKV 的固定递减模式完全不适用
4. **Vision-only compression 优于 unified**：text tokens 是"锚点"，不可压缩；破坏 text 表示导致灾难性退化
5. **Head effect 普遍存在**：少数 visual tokens 获得远高于平均的 attention，大多数则相对均匀且低——与 VL-Cache 的发现一致
6. **模型越小越敏感**：小参数模型的 token 间信息整合能力弱，更依赖保留正确的关键 visual tokens

## 局限

- **Peak memory 问题**：需在 prefill 完成后存储完整 KV cache 才能计算全局 budget——与所有层级 budget 分配方法共有
- 未探索与 KV cache quantization 的组合
- 未涉及 video VLM 中帧间冗余的利用
- α=0.9 为固定超参，未做自适应调节
- 对视觉 token 极少的场景（如单图低分辨率），收益可能有限

## 引用关系

← 直接改进：[[SnapKV]]（改进 observation window 选择）、[[PyramidKV]]（改进 budget allocation 策略）
← 比较对象：[[H2O]]、Elastic Cache、PrefixKV
← 同方向前作：[[VL-Cache]]（首篇 VLM KV cache 压缩，被引为 [33]）
→ 父概念：[[VLM KV Cache]]、[[Token Eviction]]
→ 与量化正交：可与 [[KV Cache Quantization]] 组合（尚未探索）
