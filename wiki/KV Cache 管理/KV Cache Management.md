---
title: KV Cache Management
aliases: [KV Cache 管理, KV Cache Compression, KV 缓存管理]
tags: [model-compression, concept, kv-cache, inference]
created: 2026-04-28
updated: 2026-04-28
sources: [KVQuant, KIVI, TurboQuant, H2O, SnapKV, ShadowKV, KVzip, TriAttention, StreamingLLM, Scissorhands, PyramidKV, GQA, DeepSeek-V2-MLA]
---

## 核心概念

KV Cache 是 autoregressive Transformer 推理的核心数据结构。随着序列长度和批量大小增长，KV cache 成为 GPU 显存的主要瓶颈——LLaMA-7B 在 1M context 下 KV cache 需约 128GB，远超模型权重（14GB）。

KV Cache Management 涵盖所有旨在减少 KV cache 内存占用、提升推理吞吐量的技术。本页作为**总览页**，统一组织四大子方向。

## 为什么 KV Cache 是核心瓶颈

| 场景 | 模型权重 (FP16) | KV Cache (FP16) | KV/权重 |
|------|---------------|-----------------|---------|
| LLaMA-7B, 4K context | 14GB | 0.5GB | 3.6% |
| LLaMA-7B, 128K context | 14GB | 16GB | **114%** |
| LLaMA-7B, 1M context | 14GB | 128GB | **914%** |

当 context length 超过 ~32K 时，KV cache 超过模型权重大小，成为部署的首要瓶颈。

## 四大子方向

### 一、KV Cache 量化

降低每个 KV entry 的比特数。核心发现：Key per-channel、Value per-token 量化（[[KVQuant]]、[[KIVI]] 独立验证）。

详见 **[[KV Cache Quantization]]**。

| 方法 | 会议 | 比特 | 核心技术 |
|------|------|------|---------|
| [[KVQuant]] | NeurIPS 2024 | 3-bit | NUQ + Pre-RoPE + Dense-and-Sparse |
| [[KIVI]] | ICML 2024 | 2-bit | Tuning-free 非对称均匀量化 |
| [[TurboQuant]] | ICML 2025 | 2.5-3.5 bit | Rotation + Lloyd-Max 最优量化 |

### 二、Token Eviction

移除不重要 token 的 KV 对。基于 [[Attention Sparsity]]（>95% attention 权重接近零）。

详见 **[[Token Eviction]]**。

| 方法 | 会议 | 类型 | 核心技术 |
|------|------|------|---------|
| [[H2O]] | NeurIPS 2023 | Query-aware, generation-time | Heavy Hitter + recent window |
| [[StreamingLLM]] | ICLR 2024 | Structural | [[Attention Sink]] + sliding window |
| [[Scissorhands]] | ICLR 2024 | Query-aware, generation-time | Pivotal token persistence |
| [[SnapKV]] | NeurIPS 2024 | Query-aware, prefill-time | Observation-window voting + pooling |
| [[PyramidKV]] | ACL 2025 | Layer-adaptive | 金字塔形层级预算 |
| [[KVzip]] | NeurIPS 2025 | **Query-agnostic** | Context reconstruction scoring |
| [[TriAttention]] | arXiv 2026 | **Query-agnostic** | Pre-RoPE trigonometric scoring |

### 三、Low-Rank Compression + Offloading

用低秩近似压缩 KV，结合 CPU offloading。保留全部信息但需额外传输。

详见 **[[Low-Rank KV Compression]]** 和 **[[KV Cache Offloading]]**。

| 方法 | 会议 | 核心技术 |
|------|------|---------|
| [[ShadowKV]] | arXiv 2025 | Pre-RoPE SVD + CPU offload + landmark sparse retrieval |

### 四、架构级 KV 压缩

从模型架构层面减少 KV cache 大小。

详见 **[[Grouped-Query Attention]]**。

| 方法 | 核心思路 | 压缩比 |
|------|---------|--------|
| MQA | 所有 head 共享 KV | H:1 |
| [[GQA]] | 分组共享 KV | H/G:1 |
| [[DeepSeek-V2-MLA]] | 低维潜空间投影 | 93.3% |

## 正交互补原则

四大子方向在不同维度压缩，可自由组合：

```
总 KV cache ∝ (层数) × (KV head 数) × (token 数) × (每 entry 比特数)
               架构级 GQA/MLA    Token Eviction   KV Cache 量化
                                   ↕ Low-Rank/Offload (不直接减少，而是分层存储)
```

实际组合示例：
- GQA-8 + KIVI 2-bit：架构 4× + 量化 8× = **32× 压缩**
- SnapKV 80% eviction + KIVI 4-bit + GQA-8：5× + 4× + 4× = **80× 压缩**
- KVzip 70% eviction + 4-bit 量化：3.3× + 4× = **~13× 压缩**（实测 16.3GB→1.2GB）

## Pre-RoPE 结构化特性的汇聚

多篇工作从不同角度发现了 Pre-RoPE Key 的结构化特性——这是 KV cache 管理的一个基础性发现：

| 工作 | 发现 | 利用方式 |
|------|------|---------|
| [[KVQuant]] | 通道稳定性（固定 outlier channel） | Per-channel 量化 |
| [[ShadowKV]] | 低秩性（SVD 6× 压缩） | 低秩投影存储 |
| [[TriAttention]] | 方向集中性（R > 0.95） | 三角级数 importance 预测 |

RoPE 的位置依赖旋转会破坏这些结构——这解释了为什么 Pre-RoPE 处理在量化、低秩压缩和 eviction 中都有效。

## 研究时间线

```
2019     MQA (Shazeer)                 ← 架构级：多 head 共享 KV
2023.06  H2O (NeurIPS 2023)            ← Eviction 奠基：Heavy Hitter + greedy
2023.06  GQA (EMNLP 2023)             ← 架构级：分组共享 KV，成为标准
2023.10  StreamingLLM (ICLR 2024)      ← Attention Sink 发现
2023.11  Scissorhands (ICLR 2024)      ← Pivotal token persistence
2024.01  KVQuant (NeurIPS 2024)        ← 量化：4 技术组合，3-bit 1M context
2024.02  KIVI (ICML 2024)             ← 量化：Tuning-free 2-bit
2024.04  SnapKV (NeurIPS 2024)         ← Eviction：Prefill-time voting
2024.05  DeepSeek-V2 (arXiv)          ← 架构级：MLA 潜空间投影
2024.06  PyramidKV (ACL 2025)          ← Eviction：层自适应预算
2024.12  ShadowKV (arXiv 2025)         ← Hybrid：低秩 + offload + sparse
2025.02  TurboQuant (ICML 2025)        ← 量化：理论最优 rate-distortion
2025.05  KVzip (NeurIPS 2025)          ← Eviction：Query-agnostic
2026.04  TriAttention (arXiv 2026)     ← Eviction：Pre-RoPE trigonometric
```

## 开放问题

1. **量化 × Eviction 的联合优化**：当前都是先 evict 再量化（或反之），缺乏统一优化框架
2. **GQA/MLA + 量化/eviction 的交互**：GQA 下每个 KV head 影响更多 query head，量化/eviction 误差的传播特性是否改变？
3. **多模态 LLM 的 KV cache**：视频 LLM 的 KV cache 极大（视频帧 × token），跨模态压缩是新方向
4. **1-bit KV cache 的可行性**：KIVI 做到 2-bit，理论下界在哪里？
5. **KV cache 的语义压缩**：能否不仅压缩数值，而是压缩语义——如 MLA 的潜空间投影的自动发现版本

## 引用关系

← 子概念：[[KV Cache Quantization]]、[[Token Eviction]]、[[Low-Rank KV Compression]]、[[KV Cache Offloading]]、[[Grouped-Query Attention]]
← 理论基础：[[Attention Sparsity]]、[[Attention Sink]]、[[Emergent Outlier Features]]
← 经验观测汇总：**[[KV Cache Empirical Observations]]**——系统整理所有 KV cache 研究中发现的经验性观测事实
← 全部已录入来源：见 frontmatter sources 列表
→ 与其他压缩技术正交：[[Post-Training Quantization]]（weight 量化），[[Pruning and Sparsity]]（权重剪枝）
