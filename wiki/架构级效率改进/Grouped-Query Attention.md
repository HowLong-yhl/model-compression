---
title: Grouped-Query Attention
aliases: [GQA, MQA, Multi-Query Attention, MLA, 分组查询注意力]
tags: [model-compression, concept, architecture, attention, kv-cache]
created: 2026-04-28
updated: 2026-04-28
sources: [GQA, DeepSeek-V2-MLA]
---

## 核心概念

Grouped-Query Attention (GQA) 是一种从模型架构层面减少 KV cache 大小的技术。核心思路：让多个 query head 共享同一组 Key-Value head，从而将 KV cache 大小从 H 份减少到 G 份（H 为 query head 数，G 为 KV group 数）。

GQA 已成为现代 LLM 的标准配置（LLaMA-2/3、Mistral、Gemma 均采用），是架构级 KV cache 优化的最成功实践。

## 演进线

```
MHA (Multi-Head Attention)     H 组 KV，无压缩
  ↓
MQA (Multi-Query Attention)    1 组 KV，H:1 共享，压缩最大但质量下降
  ↓
GQA (Grouped-Query Attention)  G 组 KV，H/G:1 共享，精度-压缩平衡点
  ↓
MLA (Multi-head Latent Attention)  低维投影替代 KV 共享，93.3% KV 压缩
```

### MQA（Multi-Query Attention, Shazeer 2019）

所有 query head 共享同一组 KV，KV cache 缩减为 1/H。优点是极端压缩；缺点是质量下降且从 MHA checkpoint 转换需要完整重训练。

### GQA（[[GQA|Ainslie et al., EMNLP 2023]]）

将 H 个 query head 分为 G 个组，每组共享一组 KV。关键贡献：
- 提出 **uptraining** 方法：从已有 MHA checkpoint 转换为 GQA，仅需 **5% 的原训练计算量**
- Head 合并策略：mean-pooling 原有 KV head 的权重比随机初始化更优
- G=8 时（8 组 KV），质量接近 MHA 且解码速度接近 MQA

### MLA（Multi-head Latent Attention, [[DeepSeek-V2-MLA|DeepSeek-V2 2024]]）

将 KV 投影到低维潜空间，仅缓存低维表示（而非完整 KV）。数学上：

$$c_t^{KV} = W^{DKV} h_t \quad (\text{压缩：} d \to d_c, \text{其中 } d_c \ll d)$$

推理时从低维表示重建 KV：$k_t = W^{UK} c_t^{KV}$。KV cache 从 $2 \times n_h \times d_h$ 降至 $d_c$，DeepSeek-V2 实现 **93.3% 压缩**。

**Decoupled RoPE**：RoPE 与低秩 KV 的正交处理是 MLA 的关键设计——RoPE 需要的维度单独缓存，避免破坏低秩结构（与 [[KVQuant]] 的 Pre-RoPE 发现异曲同工）。

## 与量化/eviction 的交互

GQA/MLA 是架构级压缩，与量化和 eviction 在不同层面正交：

| 维度 | 架构 (GQA/MLA) | 量化 (KIVI/KVQuant) | Eviction (H2O/SnapKV) |
|------|---------------|--------------------|-----------------------|
| 压缩对象 | KV head 数量 | 每个 entry 的比特数 | Token 数量 |
| 压缩时机 | 模型设计/训练时 | 推理时 | 推理时 |
| 典型压缩比 | 4-8× | 4-8× | 5-10× |

三者可叠加：GQA-8 + 4-bit 量化 + 50% eviction → 总压缩比 ~64×。

**注意事项**：GQA 下每个 KV head 被更多 query head 共享，因此 KV head 的量化误差会影响更多 query head 的输出——理论上 GQA 使量化更敏感。但实践中（[[KIVI]]、[[KVQuant]] 均在 GQA 模型 LLaMA-2 上验证），这一额外敏感性在 2-3 bit 下仍可控。

## 引用关系

← 已录入来源：[[GQA]]（EMNLP 2023），[[DeepSeek-V2-MLA]]（2024）
→ 概念关联：[[KV Cache Management]]（架构级 KV 压缩子方向），[[KV Cache Quantization]]（量化在 GQA 上的交互）
→ 工业影响：LLaMA-2/3、Mistral、Gemma、DeepSeek-V2 均采用 GQA 或 MLA
