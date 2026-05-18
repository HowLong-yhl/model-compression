---
title: "Attention Debiasing for Token Pruning in Vision-Language Models"
aliases: [Attention Debiasing, Zhao 2026]
source_type: paper
source_url: "https://arxiv.org/abs/2508.17830"
source_date: 2026-01-16
ingested: 2026-05-15
venue: arXiv 2026 (journal submission)
authors: [Kai Zhao, Wubang Yuan, Yuchen Lin, Liting Ruan, Xiaofeng Lu, Deng-Ping Fan, Ming-Ming Cheng, Dan Zeng]
affiliation: Shanghai University, Nankai University
code: "https://github.com/intcomp/attention-bias"
tags: [model-compression, method, kv-cache, token-pruning, VLM, vision-language, attention-bias, recency-bias, attention-sink, plug-and-play]
---

## 核心要点

本文提出了一个关键问题：**VLM 中 attention score 作为 token 重要性指标是系统性有偏的**。偏差来自 LLM 继承的两个固有缺陷——recency bias（序列末尾 token 获得不成比例的高 attention）和 attention sink（padding token 获得虚高 attention）。这两种偏差导致所有基于 attention 的 visual token pruning 方法**系统性地保留了错误的 token**（图像底部区域、padding 区域），而丢弃了语义上真正重要的视觉内容。

解决方案极为简洁：(1) 用指数函数拟合 positional bias 并归一化去除；(2) 直接将 padding token 的 attention 置零。这两个 debiasing 操作是 **model-agnostic、pruning-method-agnostic、task-agnostic** 的 plug-and-play 模块，仅需几行代码即可集成到任意基于 attention 的 VLM pruning 方法中。

![[images/AttentionDebiasing_fig1_radar.png]]
*Figure 1. 六种 pruning 方法在 10 个 VL 基准上的平均性能。红色=原始，蓝色=+Ours。Debiasing 一致提升所有方法。*

## 核心发现：Attention 的两种系统性偏差

### 1. Recency Bias（位置偏差）

VLM 继承了 LLM 的 **recency bias**——模型系统性地给序列中后出现的 token 分配更高 attention。在 VLM 中，视觉 token 按 raster scan 顺序排列（左上→右下），因此 recency bias 具体表现为：**图像底部区域的视觉 token 获得显著更高的 attention score，与语义内容无关**。

![[images/AttentionDebiasing_fig2_debiasing.png]]
*Figure 2. LLaVA-v1.5-7B 的平均 text-to-vision attention score。左：原始 attention 随 token index 指数上升（recency bias）；右：debiasing 后变为平稳的 content-dependent 信号。*

量化分析（Table VIII）：
- 在所有 4 个基准 × 4 种 pruning 方法上，原始 attention 的 bias strength $\sigma_\text{attn}$ 均为正值（0.70-1.52），表明单调递增趋势
- 更严重的是：pruning（top-K 选择）后 bias **被放大**，$\sigma_\text{freq}$ 远大于 $\sigma_\text{attn}$（如 FastV 从 0.91 放大到 2.56）
- 应用 debiasing 后两项指标均降至~0

**根因分析**：recency bias 源于 LLM 的两个固有属性——(1) 自回归训练目标使模型天然偏好 recent context；(2) RoPE 的 long-term decay 导致远距离 token 的 attention 衰减。移除 RoPE（Endo et al., ICCV 2025）并不能完全消除 bias（Table V：FastV+RoPE-free 58.4% vs FastV+Ours 62.5%），因为 recency bias 从根本上源于模型优化，而非仅仅是 positional encoding。

### 2. Padding-Induced Attention Sink

VLM 在处理非正方形图像时需要 padding 至正方形再编码。这些 **语义为空的 padding token** 却获得了异常高的 attention score——这是 LLM attention sink 现象在 VLM 中的表现。

关键发现（Figure 7）：
- padding 区域虽然语义为空，但在 hidden state 中产生了**特定维度上的极端激活值**
- 这些极端激活传播到 attention 计算中，产生虚高的 attention score
- 结果：pruning 保留了大量 padding token 而丢弃了有用的视觉 patch

![[images/AttentionDebiasing_fig7_padding_sink.png]]
*Figure 7. 不同 padding 模式下的 attention sink 可视化。Mean/Constant/Random padding 均产生 attention sink（padding 区域高 attention + 高 activation）；Mirror padding 将 sink 转移至结构化背景而非 padding 边缘。*

不同 padding 模式（Mean/Constant/Random/Mirror）的定量对比（Table IX）表明，**更换 padding 策略几乎无效**（VQAv2: 73.2/73.4/74.1/73.9）——偏差来源是 attention sink 机制本身，而非 padding 内容。唯一有效的方案是**直接移除 padding token 的 attention**（+Ours: 76.3，+2.1 绝对值）。

## 方法详解

### Positional Debiasing

将 attention score 分解为 positional bias 和 content-driven attention 的乘积：

$$a_i = b_i \cdot \hat{a}_i$$

其中 $b_i$ 是 content-agnostic 的位置偏差，$\hat{a}_i$ 是去偏后的"oracle" attention。

**Step 1**: 在大规模数据上统计 text-to-vision 平均 attention $\bar{a}_i$，用指数函数拟合平滑的 bias curve：

$$f(x) = \mu \cdot e^{\sigma x}, \quad x \in [0, 1]$$

其中 $\sigma$ 量化 recency bias 的强度，$(µ, \sigma)$ 通过最小二乘法估计。

**Step 2**: 用 fitted curve 归一化 attention：

$$\hat{a}_i = \frac{a_i}{f(i/L)}$$

指数拟合的优势：(1) 比直接用 $\bar{a}_i$ 归一化更平滑、更稳定；(2) 归一化到 $[0,1]$ 使其与 token 数量 $L$ 无关（length-agnostic）。

### Padding Attention Suppression

直接将 padding token 的 attention 置零：

$$\hat{a}_i = \begin{cases} 0, & \text{if } i \text{ is a padding token} \\ a_i, & \text{otherwise} \end{cases}$$

实践中先做 positional debiasing，再做 padding suppression。

### 集成方式

两个操作完全是 **post-hoc** 的——它们修改 pruning 方法内部的 ranking score，不改变方法本身的 pruning 逻辑、粒度或策略。几行代码即可集成到 FastV、PyramidDrop、SparseVLM、HiMAP、TokenCarve、iLLaVA 等任意 attention-based pruning 方法中。

## 实验结果

### 主结果：10 个 Image 基准

![[images/AttentionDebiasing_table2_main_results.png]]
*Table II. 10 个 VL 基准上 128 visual token budget 的定量对比。Debiasing 一致提升所有 6 种 attention-based pruning 方法。*

**LLaVA-v1.5-7B** (budget=128 visual tokens)：
- **FastV**: 59.6 → 62.5 (**+2.9**)——最大提升，因为 FastV 使用 output-to-visual attention 单层 pruning，最受 recency bias 影响
- **HiMAP**: 59.0 → 61.4 (**+2.4**)
- **iLLaVA**: 59.7 → 61.7 (**+2.0**)
- **PyramidDrop**: 60.5 → 61.6 (**+1.1**)
- **TokenCarve**: 61.5 → 61.9 (**+0.4**)
- **SparseVLM**: 62.0 → 62.4 (**+0.4**)——提升最小，因为 SparseVLM 用 selected text tokens 的 attention（本身部分缓解了 bias）

**LLaVA-v1.5-13B** 上趋势一致，最大提升 iLLaVA +1.0，HiMAP +1.0，FastV +0.9。

对比 non-attention-based 方法（ToMe, PruMerge+, VisionZip）：debiased FastV (62.5) 已超过所有 non-attention 方法，接近 full model (63.1)。

### 定性对比

![[images/AttentionDebiasing_fig3_qualitative.png]]
*Figure 3. 四种 pruning 方法 before/after debiasing。原始方法保留 padding 和底部区域 token，丢弃关键视觉内容（数字、球员编号等）；debiasing 后保留语义相关 patch，回答正确。*

典型案例：
- **SparseVLM** 问"尺子上的数字"→ 原始保留 padding 区域 token、丢弃数字 "2,3,4" 所在 patch → 回答"0"；+Ours 正确保留数字 patch → "2,3,4"
- **HiMAP** 问"投手号码"→ 原始回答"21"（错误），+Ours 正确回答"23"

### Video QA

在 Video-LLaVA 上测试（2048→512 video tokens），3 个视频 QA 基准：
- SparseVLM: 57.6 → 58.2 (+0.6 avg)
- FastV: 57.3 → 57.9 (+0.6)
- PyramidDrop: 57.8 → 57.9 (+0.1)

增益相对较小——视频帧通常已是正方形，padding 较少。

## 消融实验

| 组件 | VQAv2 | GQA | TextVQA | Avg |
|------|:---:|:---:|:---:|:---:|
| FastV baseline | 73.2 | 55.8 | 56.9 | 62.0 |
| +PD only | 76.0 | 58.3 | 57.7 | 64.0 |
| +PAS only | 76.3 | 58.8 | 57.6 | 64.2 |
| +PD+PAS (Ours) | **76.6** | **59.3** | **58.0** | **64.6** |

| 组件 | VQAv2 | GQA | TextVQA | Avg |
|------|:---:|:---:|:---:|:---:|
| PyramidDrop baseline | 75.0 | 57.0 | 56.1 | 62.7 |
| +PD only | 76.1 | 58.4 | 56.2 | 63.6 |
| +PAS only | 76.2 | 58.7 | 56.4 | 63.8 |
| +PD+PAS (Ours) | **76.5** | **58.9** | **56.7** | **64.0** |

两个组件**独立有效且互补**。Padding suppression 单独比 positional debiasing 略强（在 FastV 上 64.2 vs 64.0），但组合效果最好。

### 鲁棒性

- **Token budget 敏感性**：budget 越小（更激进 pruning），debiasing 增益越大——budget=64 时增益最大（TextVQA: +2.0），budget=256 时增益最小（+0.6）
- **Pruning layer 选择**：在 layer 1-16 上均有提升，越早 pruning（layer 1-2）增益越大
- **RoPE 方案对比**：直接移除 RoPE（Endo et al.）反而降低性能（FastV: 59.6 → 58.4），因为 RoPE 提供有用的位置信息，全移除得不偿失。Debiasing 保留位置信息同时去除偏差。

## 研究意义

### 对 VLM KV Cache / Token Pruning 领域的启示

1. **Attention score 是有偏估计**：这是对所有 attention-based VLM pruning/eviction 方法的通用警告。[[AirCache]] 用 elite text tokens 缓解 observer 不一致性，[[MixKV]] 用 diversity 补充 importance-only 的盲区——本文揭示了第三种偏差源：**位置偏差和 padding 偏差**。三者从不同角度说明 raw attention score 不可直接用于 VLM token 选择。

2. **Token pruning vs KV cache eviction 的桥梁**：本文的 debiasing 虽然针对 token pruning（prefill 阶段丢弃 visual token），但 recency bias 和 attention sink 同样影响 KV cache eviction 中的 attention-based scoring（如 [[SnapKV]]、[[H2O]]）。原则上，positional debiasing 可以直接应用于 KV cache eviction 的评分环节。

3. **Plug-and-play 的增强范式**：与 [[MixKV]] 类似，本文不是提出新的 pruning 方法，而是提出一种可插拔的 scoring 改进。这种"增强而非替代"的范式在 VLM 效率领域越来越常见。

### 与 TriAttention 的有趣对比

[[TriAttention]] 从 pre-RoPE 空间绕开了 post-RoPE 的位置偏差问题（用三角级数理论做 query-agnostic scoring），本文则选择直接在 post-RoPE 的 attention score 上做统计校正。两种策略对 recency bias 的处理殊途同归，但出发点不同：TriAttention 是"理论驱动的绕行"，Attention Debiasing 是"经验驱动的修正"。

## 局限性

- **仅限 attention-based pruning**：对 non-attention 方法（ToMe, VisionZip, PruMerge+）无法增强
- **Padding suppression 的前提**：需要 encoder 显式引入 padding——padding-free 或 dynamic resolution 的新架构（如 Qwen2-VL 的 naive dynamic resolution）不适用
- **推理时修正而非根治**：recency bias 和 attention sink 的根因在 LLM 的训练目标和优化特性中，debiasing 是 inference-time 的缓解措施
- **仅在 LLaVA 架构上验证**：未测试 InternVL、Qwen-VL 等其他 VLM 架构
- **指数拟合的假设**：隐式假设 recency bias 遵循指数形式——更复杂的 bias pattern 可能需要其他函数族

## 引用关系

← 研究问题：VLM 中 attention-based token pruning 的 attention bias 问题
← 背景知识：Recency bias in LLMs (Liu et al. 2024, "Lost in the Middle"), Attention Sink (Xiao et al. 2024, [[StreamingLLM]]), Visual Attention Sink (Kang et al. 2025)
← 增强对象：FastV, PyramidDrop, SparseVLM, HiMAP, TokenCarve, iLLaVA
← 对比方法：RoPE-free scoring (Endo et al., ICCV 2025), VisPruner (Zhang et al., ICCV 2025)
→ 贡献：首次系统识别并定量化 VLM attention 中的 recency bias + padding attention sink 对 token pruning 的影响，提出轻量 plug-and-play debiasing
→ 研究启示：raw attention score 在 VLM 中是有偏估计——位置偏差 ([[Attention Debiasing]])、observer 不一致性 ([[AirCache]])、语义覆盖不足 ([[MixKV]]) 是三个独立的偏差源
→ 概念关联：[[Token Eviction]]、[[VLM KV Cache]]、[[Attention Sparsity]]
