---
title: "ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference"
aliases: [ShadowKV, Sun 2025]
source_type: paper
source_url: "https://arxiv.org/abs/2412.06015"
source_date: 2024-12-08
ingested: 2026-04-28
venue: arXiv 2025
authors: [Hanshi Sun, Li-Wen Chang, Wenlei Bao, Size Zheng, Ningxin Zheng, Xin Liu, Harry Dong, Yuejie Chi, Beidi Chen]
affiliation: ByteDance Seed, Carnegie Mellon University
code: "https://github.com/ByteDance-Seed/ShadowKV"
tags: [model-compression, method, kv-cache, offload, low-rank, sparse-attention, long-context, inference]
---

## 核心要点

ShadowKV 是一个面向超长上下文（128K-1M token）高吞吐推理的 KV cache 管理系统，采用**混合策略**——不做 eviction（保留所有 KV），而是通过低秩压缩 + CPU offload + 稀疏检索的组合来高效管理 KV cache。

两个核心发现：
1. **Pre-RoPE Key 极度低秩**：RoPE 之前的 Key cache 可通过截断 SVD 压缩 **6× 以上**几乎无损——这是因为同一序列内的 pre-RoPE key 共享低秩子空间
2. **Post-RoPE Key 的空间局部性**：相邻 token 的 post-RoPE key 余弦相似度极高，允许 chunk-level 近似

ShadowKV 在 A100 上实现了等效 **7.2 TB/s** 的带宽（原生仅 2 TB/s），支持 6× batch size 提升和 3.04× 吞吐改善。

## 方法详解

### Pre-RoPE Key 的低秩性

**关键发现**：对 LLaMA-3-8B-1M 的 Key cache 进行 SVD 分析：
- Pre-RoPE Key 的有效秩极低——前 25% 的奇异值即可保留 >99% 的 Frobenius 范数
- Post-RoPE Key 的有效秩显著更高——RoPE 的位置依赖旋转破坏了低秩结构
- 这与 [[KVQuant]] 的 Pre-RoPE 发现一致：RoPE 会破坏 Key 的结构化特性

**与权重低秩的区别**：权重矩阵的低秩结构是模型固有的（与输入无关），可通过静态权重分解处理。Pre-RoPE Key 的低秩是**数据依赖的**——不同序列的低秩子空间不同，必须在线计算 SVD。

### 系统架构

#### Prefill 阶段

1. **Key 低秩压缩**：对每层的 pre-RoPE Key cache $K \in \mathbb{R}^{n \times d}$ 做截断 SVD，仅保留低秩投影 $(A, B)$，满足 $K \approx A B^T$。$A, B$ 存储在 GPU 上
2. **Landmark 计算**：将 post-RoPE Key 按 chunk 分段（chunk size 如 64），计算每个 chunk 的均值作为 landmark 存储在 GPU 上
3. **Outlier chunk 检测**：chunk 内 token 间最小余弦相似度低于阈值的 chunk 标记为 outlier，其完整 KV 保留在 GPU 上作为 static cache
4. **Value offload**：完整的 Value cache offload 到 CPU 内存

#### Decoding 阶段

每步解码：
1. **Landmark 检索**：用当前 query 与所有 chunk landmark 计算近似 attention score，选出 top-k chunk
2. **Key 重建（GPU）**：从低秩投影 $(A, B)$ 重建选中 chunk 的 pre-RoPE Key，再施加 RoPE
3. **Value 取回（CPU→GPU）**：通过 PCIe 从 CPU 取回选中 chunk 的 Value
4. **流水线并行**：Key 重建和 Value 取回用 CUDA multi-stream 并行，隐藏 PCIe 延迟

#### Cache Hit 策略

利用解码步之间 KV 选择的**时间局部性**——相邻解码步通常选择高度重叠的 chunk。在 GPU 维护一个小的 Value 缓冲区，优先命中已在 GPU 上的 chunk，减少 ~60% 的 PCIe 数据传输。

## 实验结果

### 精度（RULER, 128K context）

| 模型 | 方法 | RULER Avg. |
|------|------|-----------|
| LLaMA-3-8B-1M | Full Attention | 86.68 |
| | **ShadowKV** | **86.88** |
| | Quest | 82.03-83.99 |
| | InfiniGen | 70.13-78.31 |
| | SnapKV | — (OOM at 128K) |

ShadowKV 达到甚至略超 Full Attention，远优于其他 sparse attention 方法。

### 系统性能（A100）

| 指标 | 效果 |
|------|------|
| Batch size 扩大 | **6×** |
| 吞吐提升 (LLaMA-3.1-8B, 122K) | **3.04×** (245.90 vs 80.78 tokens/s) |
| 等效带宽 | **7.2 TB/s** (vs 2 TB/s native) |
| 稀疏预算 | 仅 **1.56%** 的 KV 对 |

### 多轮对话（Multi-turn NIAH）

ShadowKV 在 8 轮对话后仍保持检索精度，而 SnapKV 在第 1 轮后即退化——因为 ShadowKV 保留了全部 KV 信息（只是分散存储），不存在信息丢失。

## 局限性

- **需要 CPU 内存**：Value cache 全量 offload 到 CPU，需要充足的 host memory
- **PCIe 带宽依赖**：系统性能受限于 GPU-CPU 互联带宽，在 PCIe Gen3 系统上收益更小
- **在线 SVD 开销**：每个序列需要在 prefill 阶段计算 SVD，增加首 token 延迟
- **Chunk size 是超参数**：过小则 landmark 不够稳定，过大则稀疏检索粒度不够

## 与已有知识的关联

### Pre-RoPE 发现的汇聚

ShadowKV 的 pre-RoPE Key 低秩发现与 [[KVQuant]] 的 Pre-RoPE Key Quantization 和 [[TriAttention]] 的 Pre-RoPE Q/K Concentration 从不同角度验证了同一个基础事实：**RoPE 前的 Key 具有更强的结构化特性**（低秩、通道稳定、方向集中），RoPE 的位置依赖旋转会破坏这些特性。

### 不同于 Eviction 的哲学

与 [[H2O]]、[[SnapKV]]、[[KVzip]] 等 eviction 方法的根本区别：ShadowKV **不丢弃任何信息**——所有 KV 对都保留在 CPU 上，通过稀疏检索按需取回。这避免了 eviction 方法的不可恢复性问题，代价是需要额外的 CPU 内存和 PCIe 带宽。

### 低秩 + 稀疏的组合

ShadowKV 的低秩 Key + 稀疏 Value 检索策略与量化领域的 GEAR（低秩 + 稀疏残差）在思路上异曲同工——都利用了"低秩捕捉主体信号 + 稀疏处理残余极端值"的互补组合。

## 引用关系

← 相关发现：[[KVQuant]]（Pre-RoPE Key 量化发现），[[TriAttention]]（Pre-RoPE Q/K concentration）
← 技术基础：Truncated SVD, [[Attention Sparsity]]（稀疏检索的可行性）
→ 贡献：首个系统化的低秩 + offload + sparse 三位一体 KV cache 管理方案
→ 概念关联：[[Low-Rank KV Compression]]，[[KV Cache Offloading]]，[[KV Cache Management]]
→ 与 eviction/量化正交：可进一步对 GPU 上的 static cache 做量化
