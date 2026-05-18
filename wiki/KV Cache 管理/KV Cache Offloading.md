---
title: KV Cache Offloading
aliases: [KV Offload, CPU Offloading, KV Cache CPU Offload]
tags: [model-compression, concept, kv-cache, system, offload]
created: 2026-04-28
updated: 2026-04-28
sources: [ShadowKV]
---

## 核心概念

KV Cache Offloading 将 KV cache（或其一部分）从 GPU 显存转移到 CPU 内存或 NVMe 存储，通过 PCIe/NVLink 按需取回。核心优势：GPU 显存释放给 batch 并行，而 CPU 内存（通常 256GB-1TB）远大于 GPU 显存（40-80GB）。

与 eviction 的根本区别：offloading **不丢弃任何信息**——所有 KV 对都保留在系统中，代价是额外的数据传输延迟。

## 关键挑战

**PCIe 带宽瓶颈**：GPU-CPU 互联带宽（PCIe Gen4 ~32 GB/s，Gen5 ~64 GB/s）远低于 GPU 显存带宽（A100 HBM: 2 TB/s）。如果每步解码都需要取回大量 KV，offload 的收益会被 PCIe 延迟抵消。

**解决思路**：
- **稀疏检索**（[[ShadowKV]]）：每步仅取回 1-2% 的 KV（通过 landmark-based top-k 选择），将 PCIe 数据量控制在可接受范围
- **流水线并行**：Key 重建（GPU 计算）与 Value 取回（PCIe 传输）并行执行，隐藏延迟
- **Cache hit 策略**：利用解码步之间 KV 选择的时间局部性，在 GPU 维护小缓冲区减少重复传输

## ShadowKV 的实践

[[ShadowKV]] 通过三个技术组合实现了等效 **7.2 TB/s** 的带宽（原生 A100 仅 2 TB/s）：

1. Key 低秩投影存 GPU（小内存，O(r×d)），Value 全量 offload 到 CPU
2. Landmark（chunk 均值）存 GPU，用于稀疏检索选择 top-k chunk
3. CUDA multi-stream 并行 + cache hit 策略减少 ~60% PCIe 传输

## 引用关系

← 已录入来源：[[ShadowKV]]（低秩 + offload + sparse 三位一体方案）
← 相关系统：FlexGen（early offloading work），InfiniGen
→ 概念关联：[[KV Cache Management]]，[[Low-Rank KV Compression]]（常与低秩压缩配合）
→ 与其他方法正交：offload 保留全部信息，可在 GPU 端的 cache 上叠加量化或 eviction
