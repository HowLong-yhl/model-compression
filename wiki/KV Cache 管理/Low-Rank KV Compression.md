---
title: Low-Rank KV Compression
aliases: [KV 低秩压缩, SVD KV Compression, Low-Rank Key Compression]
tags: [model-compression, concept, kv-cache, low-rank]
created: 2026-04-28
updated: 2026-04-28
sources: [ShadowKV]
---

## 核心概念

Low-Rank KV Compression 通过低秩近似（如截断 SVD）压缩 KV cache，利用 KV 矩阵的低秩结构存储紧凑的投影表示。与 eviction（丢弃 token）和量化（降低比特数）不同，低秩压缩**保留所有 token 的信息**，但以低维近似替代原始表示。

## 核心发现：Pre-RoPE Key 的低秩性

[[ShadowKV]] 系统化地发现 Pre-RoPE Key cache 具有极低的有效秩——前 25% 的奇异值即可保留 >99% 的 Frobenius 范数，实现 **6× 以上无损压缩**。

关键细节：
- **Pre-RoPE vs Post-RoPE**：RoPE 的位置依赖旋转显著增加了 Key 的有效秩，Pre-RoPE Key 的低秩性远好于 Post-RoPE
- **数据依赖性**：不同序列的低秩子空间不同，必须在线计算 SVD（不能用静态权重分解替代）
- **Value 不低秩**：Value cache 的有效秩明显高于 Key，不适合低秩压缩

这一发现与 [[KVQuant]] 的 Pre-RoPE Key Quantization（通道稳定性）和 [[TriAttention]] 的 Pre-RoPE Q/K Concentration（方向集中性）从不同角度验证了同一基础事实：RoPE 前的 Key 具有更强的结构化特性。

## 与权重低秩压缩的区别

| 维度 | 权重低秩（LoRA 类） | KV 低秩（ShadowKV 类） |
|------|-------------------|----------------------|
| 目标 | 模型参数 | 推理时的激活缓存 |
| 低秩来源 | 模型固有 | 数据依赖 |
| 计算时机 | 离线（训练时/部署前） | 在线（每个序列的 prefill 阶段） |
| SVD 开销 | 一次性 | 每个序列一次 |

## 与 MLA 的关系

[[DeepSeek-V2-MLA]] 的 Multi-head Latent Attention 也是低秩 KV 压缩，但在架构层面实现——训练时就将 KV 投影到低维潜空间。ShadowKV 是 post-hoc 的低秩压缩——对已训练好的模型在推理时做 SVD。两者在数学上相似但工程实现完全不同。

## 引用关系

← 已录入来源：[[ShadowKV]]（在线 SVD Key 压缩）
← 相关技术：[[DeepSeek-V2-MLA]]（架构级低维投影），GEAR（低秩残差补偿）
→ 概念关联：[[KV Cache Management]]（KV 管理四大子方向之一），[[KV Cache Offloading]]（常与 offload 组合使用）
