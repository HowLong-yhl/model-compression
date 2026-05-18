---
title: VLM KV Cache 管理
aliases: [VLM KV Cache, Vision-Language Model KV Cache, 多模态 KV Cache]
tags: [model-compression, kv-cache, VLM, vision-language, modality-aware, concept]
created: 2026-05-14
updated: 2026-05-14
sources: [VL-Cache, AirCache, MixKV, Attention Debiasing]
---

## 为何单独设立 VLM KV Cache 专题

Vision-Language Model (VLM) 的 KV Cache 管理与纯 LLM 存在根本性差异，不能简单套用 LLM 的方法。核心区别来自 **多模态 prompt 结构**——VLM 的输入由视觉 token（image patches）和语言 token 交织组成，这导致了 LLM 中不存在的特殊现象和优化空间。

### LLM vs VLM KV Cache 的核心差异

| 维度 | LLM | VLM |
|------|-----|-----|
| Token 类型 | 单一（语言） | 混合（vision + language），存在 modality boundary |
| Attention 模式 | 整体稀疏，无结构性分界 | 视觉 token 间近乎均匀 attend；语言→视觉高度集中 |
| Sparsity 特性 | 全局性的 attention sparsity | Modality-dependent sparsity：视觉区域稠密，语言区域更稀疏 |
| Token 冗余来源 | 语义冗余（重复/不重要的语言 token） | 视觉 patch 冗余（大量背景 patch 几乎不被引用） |
| Token 重要性评估 | Accumulated attention / sliding window | 需区分 modality——post-vision attention 是更好的 proxy |
| 压缩潜力 | 10-30% cache 通常足够 | VLM 可能更极端：10% cache 即 98%+ 精度（视觉冗余极高） |
| 典型 prompt 长度 | 数百到数万 token | 视觉编码后通常数千 token（如 576 patch tokens/image），多图场景可达数万 |

### 关键发现：Modality Boundary

[[VL-Cache]] 首次系统性地揭示了 VLM attention 中的 **modality boundary 现象**：

1. **Visual tokens 之间的 self-attention** 近乎均匀分布——这意味着传统基于 attention score 差异的 eviction 方法在视觉区域缺乏区分度
2. **Post-vision language tokens 对 visual tokens 的 attention** 高度集中于少数关键 patch——这提供了真正的重要性信号
3. **Post-vision language tokens 的 attention pattern 与 decoding token 高度一致**（Pearson 相关 0.695），而与 visual tokens 的 self-attention 模式截然不同
4. **Layer-wise sparsity 非单调**：首 2 层远比其他层稠密，不符合 PyramidKV 假设的线性递减

这些发现意味着 LLM 的 eviction 策略（如 [[H2O]]、[[SnapKV]]、[[StreamingLLM]]）直接用于 VLM 会产生次优结果——它们无法感知 modality boundary，可能错误保留大量冗余的视觉 token 或错误删除少量但关键的视觉 token。

## VLM KV Cache 管理的独特挑战

### 1. 视觉 Token 的特殊性

VLM 中视觉 token 占据 prompt 的主要部分（如 LLaVA 中 576 patch tokens/image），但信息密度极不均匀：背景 patch 提供极少语义信息，前景关键 patch 则承载几乎全部视觉信息。这与语言 token 的"均匀重要"假设不同——语言中每个 token 都可能是关键名词/动词，但视觉 patch 中可能只有 5-10% 真正重要。

### 2. Modality-Aware 评分

传统 attention-based scoring（如 H2O 的 Accumulated Attention）对 VLM 有偏差——视觉区域内的均匀 attention 让所有 patch 看起来"同等重要"。VL-Cache 提出的 **Accumulated Post-vision Attention** 通过仅计算语言 token 对视觉 token 的注意力分布来评分，有效绕过了视觉 self-attention 的"均匀陷阱"。

### 3. Layer-Adaptive Budget 的非单调性

VLM 中 layer-wise sparsity 呈非单调分布（非 PyramidKV 假设的线性递减），且分布 prompt-dependent——不同的图文组合产生不同的 per-layer 稀疏度分布。这要求 budget allocation 必须是 **per-prompt 自适应**的，而非预设固定规则。

### 4. 多图/视频场景的特殊考虑

多图或视频 VLM 面临额外挑战：
- **跨图/跨帧冗余**：不同图片的背景 patch 之间可能高度冗余
- **时序信息保持**：视频帧之间需要保持时序连续性，eviction 策略需 temporal-aware
- **Token 规模爆炸**：多图/视频的 visual token 可达数万甚至十万，使 KV cache 压缩更为紧迫

## 与 LLM KV Cache 管理的关系

VLM KV Cache 管理并非完全独立于 LLM 方法——它继承了 LLM 的基本框架（token eviction、quantization、budget allocation），但在每个环节都需要 modality-aware 的适配：

| LLM 方法 | VLM 适配需求 |
|----------|-------------|
| [[H2O]] (accumulated attention) | 需改为 post-vision attention scoring |
| [[PyramidKV]] (固定递减 budget) | 需改为 prompt-dependent 非单调分配 |
| [[DuoAttention]] (head-level 分类) | 可扩展为 modality × head 双维度分类 |
| [[KV Cache Quantization]] | 可能需要 vision/language token 差异化量化精度 |
| [[StreamingLLM]] (sink + recent) | 在 VLM 中 visual tokens 不适合被当作"recent"驱逐 |

## 方法演进

```
2024.10  VL-Cache (arXiv 2024)        ← 首篇 VLM 专用 KV cache 压缩：modality boundary 发现 + post-vision attention scoring + layer-adaptive sparsity-aware budget
2025.01  MixKV (SJTU, ICLR 2026)       ← Importance+Diversity 联合优化：head-wise 语义冗余度自适应平衡，plug-and-play 框架，极端压缩(budget=64)平均+5.1%
2025.07  AirCache (Alibaba, arXiv 2025) ← Elite observation window + strength/skewness budget allocation，跨架构验证（LLaVA-OV/InternVL2/Qwen2-VL），1% cache 下比 SnapKV 高 6.5%
2026.01  Attention Debiasing (Shanghai U/Nankai, arXiv 2026) ← 系统揭示 recency bias + padding attention sink，指数去偏 + padding 置零，plug-and-play 增强 6 种 pruning 方法
```

该方向快速发展中。后续发展方向包括：

- **Modality-aware quantization**：视觉 token 和语言 token 使用不同量化精度（如视觉 2-bit、语言 4-bit）
- **Cross-image/video deduplication**：多图/视频场景下跨帧视觉 token 去重
- **Vision encoder 联合优化**：在 vision encoder 端即减少冗余 patch 输出（token merging/pruning before LLM）
- **与 VLM weight quantization 的联合**：[[Q-VLM]] 做 weight quantization，VL-Cache 做 KV cache eviction，两者可组合
- **Retrieval + Eviction 混合**：对重要视觉 token 不 evict 而是 offload，需要时 retrieve

## 已录入方法

| 方法 | 时间 | 核心创新 | 压缩效果 |
|------|------|---------|----------|
| [[VL-Cache]] | 2024.10 | Modality boundary + post-vision attention scoring + sparsity-aware layer-adaptive budget | 10% cache → 98%+ 精度，7.08× decode speedup |
| [[MixKV]] | 2025.01 | Importance+Diversity 联合：head-wise 冗余度自适应混合，plug-and-play | Budget=64 平均 +5.1%，GUI +8-9% |
| [[AirCache]] | 2025.07 | Elite observation window (self-attention 筛选关键 text tokens) + strength/skewness budget allocation | 10% cache → <1% 精度损失，29%-66% decode latency 降低 |
| [[Attention Debiasing]] | 2026.01 | Recency bias 指数去偏 + padding attention sink 置零，plug-and-play 增强任意 attention-based pruning | FastV +2.9, HiMAP +2.4 avg across 10 benchmarks (128 tokens) |

## 开放问题

1. **VLM 量化+Eviction 联合**：两篇工作均仅做 eviction，未探索与 KV quantization 的组合——视觉和语言 token 是否需要不同量化精度？
2. **更广泛的 VLM 验证**：AirCache 已扩展到 InternVL2、Qwen2-VL，验证 modality boundary 和 visual token redundancy 是跨架构普遍现象
3. **Video VLM**：视频理解模型中帧间冗余如何利用？AirCache 在 MMBench-Video 上初步验证但未深入探索 temporal-aware eviction
4. **动态多模态交互**：交错的多轮图文对话（multi-turn multi-image）中，earlier images 的 KV cache 如何管理？
5. **与 Vision Encoder 端压缩的协同**：FastV、LLaVA-PruMerge 等在 vision encoder 端就减少 token 输出，这与 KV cache 端 eviction 如何协同？AirCache 已验证优于 FastV/IVTP 等 token pruning 方法。[[Attention Debiasing]] 从 attention bias 角度桥接两类方法——其 debiasing 同时适用于 token pruning 和 KV cache eviction 的 attention-based scoring
6. **Peak Memory 问题**：层级 budget 分配方法都需要 prefill 后暂存完整 KV cache——如何在保持自适应分配的同时降低 peak memory？
7. **Attention 作为重要性指标的可靠性**：多篇工作从不同角度揭示 raw attention score 的不足——recency bias + padding sink ([[Attention Debiasing]])、observer 不一致性 ([[AirCache]])、语义覆盖不足 ([[MixKV]])、value norm 被忽略（Guo et al. EMNLP 2024）。是否需要一个更统一的 importance 框架？

## 引用关系

← 理论基础：[[Attention Sparsity]]（VLM 中同样存在稀疏性，但分布不同于 LLM）
← 父概念：[[Token Eviction]]（VLM KV Cache eviction 是 Token Eviction 的 modality-aware 特化）
← 已录入方法：[[VL-Cache]]、[[AirCache]]、[[MixKV]]、[[Attention Debiasing]]
→ 与量化正交：可与 [[KV Cache Quantization]] 组合（VLM-specific 量化尚待探索）
→ 相关 VLM 量化：[[Q-VLM]]（VLM weight quantization）、[[VLM Quantization Best Practices]]
→ LLM 方法参考：[[H2O]]、[[SnapKV]]、[[PyramidKV]]、[[DuoAttention]]、[[StreamingLLM]]
