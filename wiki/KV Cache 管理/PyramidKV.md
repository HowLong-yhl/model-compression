---
title: "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling"
aliases: [PyramidKV, Cai 2024]
source_type: paper
source_url: "https://arxiv.org/abs/2406.02069"
source_date: 2024-06-04
ingested: 2026-04-28
venue: ACL 2025
authors: [Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Tianyu Pang, Guansong Lu, Enhong Chen, Hui Xue, Zhewei Wei]
affiliation: Renmin University of China, Zhejiang University, Sea AI Lab, Ant Group, University of Science and Technology of China
code: "https://github.com/Zefan-Cai/PyramidKV"
tags: [model-compression, kv-cache, layer-adaptive, attention-sparsity, token-eviction, long-context, inference-efficiency]
---

## 核心要点

PyramidKV 的核心贡献在于揭示了 LLM 不同层的注意力模式存在 **系统性的层级差异**，并据此提出了 **层级自适应（layer-adaptive）** 的 KV cache 预算分配策略。作者通过对 LLaMA 和 Mistral 等模型的注意力模式进行可视化分析，发现了一个关键现象：**底层（lower layers）的注意力分布更分散**，关注更多的 token；**高层（upper layers）的注意力分布更集中**，仅聚焦于少量关键 token。

基于这一 **Pyramidal Information Funneling（金字塔信息漏斗）** 现象，PyramidKV 为不同层分配不同大小的 KV cache 预算：底层保留更多 KV 对，高层保留更少 KV 对，形成金字塔形状的 cache 分配。这种方法在仅使用全量 KV cache **12%** 的情况下，在 LongBench 等长文本基准测试上保持了与全量 cache 相当甚至相近的性能。

## 方法详解

### 注意力模式的层级差异

作者通过分析多种 LLM 的注意力矩阵，将不同层的注意力模式分为三类：

1. **底层（Lower Layers）-- 全局模式**：
   - 注意力分布相对均匀，关注大量 token
   - 注意力矩阵接近"全亮"（dense pattern）
   - 需要较大的 KV cache 来保留足够的信息

2. **中间层（Middle Layers）-- 过渡模式**：
   - 注意力开始聚焦，但仍保持中等程度的分散
   - 需要适中的 KV cache 预算

3. **高层（Upper Layers）-- 稀疏/集中模式**：
   - 注意力高度集中于少数关键 token（如 attention sink 和局部 token）
   - 注意力矩阵呈现明显的稀疏"列"模式
   - 仅需较小的 KV cache 即可捕获关键信息

### 金字塔形 Cache 分配策略

PyramidKV 的核心算法：

```
给定总 KV cache 预算 C 和模型层数 L：

1. 将层按深度分为若干组（如 bottom/middle/top）
2. 按金字塔比例分配预算：
   - Bottom layers: 每层分配 C_bottom = α × C_avg（α > 1）
   - Middle layers: 每层分配 C_middle = C_avg
   - Top layers:    每层分配 C_top = β × C_avg（β < 1）
   约束：所有层的预算总和 = C

3. 在每一层内，使用 attention score 选择 top-k 个最重要的 token
   保留其 KV 对，驱逐其余 token
```

### 预算分配的具体形状

作者对比了多种分配形状：

| 形状 | 描述 | 底层 : 中层 : 高层 |
|------|------|-------------------|
| **Pyramid（金字塔）** | 底多高少 | 大 : 中 : 小 |
| Hourglass（沙漏） | 两端多中间少 | 大 : 小 : 大 |
| Uniform（均匀） | 所有层相同 | 等 : 等 : 等 |
| Inverse Pyramid | 底少高多 | 小 : 中 : 大 |

实验验证 **Pyramid 形状** 在大多数任务上表现最优，与注意力分布的层级差异观察一致。

### Token 选择策略

在每一层确定了 cache 预算后，PyramidKV 使用以下策略选择保留的 token：

1. **基于 Prefill 阶段注意力**：利用 prompt 处理阶段（prefill）最后一个 token（或最后几个 token 组成的观察窗口）的注意力分数来评估 token 重要性。
2. **Per-layer 独立选择**：每层独立执行 token 选择，不同层可以保留不同的 token 子集。
3. **保留 sink tokens 和 local tokens**：与 [[StreamingLLM]] 类似，始终保留初始 sink tokens 和最近的 local window tokens。

### 自适应预算计算

PyramidKV 还支持根据输入自适应计算预算分配：
- 对每一层计算注意力分布的**熵（entropy）**
- 高熵层（注意力分散）-> 分配更多预算
- 低熵层（注意力集中）-> 分配更少预算
- 这使得预算分配能够根据具体输入动态调整

## 实验结果

### LongBench 基准测试

| 方法 | KV Cache 比例 | 平均分数 | 相对全量 Cache |
|------|-------------|---------|--------------|
| Full KV Cache | 100% | 42.1 | 基线 |
| SnapKV (均匀分配) | 12% | 39.8 | -5.5% |
| H2O | 12% | 37.2 | -11.6% |
| StreamingLLM | 12% | 31.5 | -25.2% |
| **PyramidKV** | **12%** | **40.9** | **-2.9%** |

### 不同 Cache 预算下的表现

| KV Cache 预算 | PyramidKV | SnapKV | H2O |
|--------------|-----------|--------|-----|
| 64 per layer (极端) | 35.2 | 31.8 | 28.5 |
| 128 per layer | 38.9 | 36.5 | 33.2 |
| 256 per layer | 40.9 | 39.8 | 37.2 |
| 512 per layer | 41.7 | 41.2 | 39.8 |

### 关键发现

- 在极端低预算（如每层仅 64 个 token）下，PyramidKV 相对于均匀分配策略的优势最为显著（提升 3-5 个百分点）。
- 随着预算增大，金字塔分配与均匀分配的差距缩小，因为所有层都有足够的 cache。
- 在 Mistral-7B-Instruct 和 LLaMA-3-8B-Instruct 上均验证了一致的趋势。

### 分配形状对比

| 形状 | LongBench 平均 | RULER 平均 |
|------|--------------|-----------|
| Uniform | 39.8 | 72.1 |
| Hourglass | 38.5 | 70.3 |
| Inverse Pyramid | 37.2 | 68.8 |
| **Pyramid** | **40.9** | **74.5** |

## 局限性

1. **预算分配比例需要经验设定**：虽然金字塔形状是最优的，但具体的底层/中层/高层预算比例仍需根据模型和任务进行调优。
2. **依赖 Prefill 阶段的注意力信号**：token selection 基于 prefill 阶段的注意力，对于 prompt 较短的场景，注意力信号可能不够稳定。
3. **与模型架构耦合**：金字塔观察基于特定的模型（LLaMA/Mistral），对于其他架构（如 Mamba、RWKV 等非 Transformer 架构），该观察可能不成立。
4. **一次性选择的局限**：PyramidKV 在 prefill 后一次性决定保留哪些 token，在长文本生成过程中不会动态调整，可能不适应 attention 模式随生成进展而变化的场景。
5. **额外的分析开销**：自适应预算计算需要对每一层计算注意力熵，增加了少量 prefill 阶段的延迟。

## 与已有知识的关联

- **与 [[SnapKV]] 的关系**：PyramidKV 可以被视为 SnapKV 的 **层级感知增强版**。SnapKV 对所有层使用相同的 cache 预算（均匀分配），而 PyramidKV 根据层级差异进行自适应分配。两者在 token 选择策略上相似（都基于 prefill 注意力），关键区别在于 **预算分配策略**。
- **与 [[StreamingLLM]] 的关系**：PyramidKV 同样保留 sink tokens，并受到 [[Attention Sink]] 现象的启发。但 StreamingLLM 面向流式推理场景，而 PyramidKV 面向有限长度的 prompt 压缩。
- **与 [[H2O]] 的关系**：H2O 在 decoding 阶段动态驱逐 token，而 PyramidKV 在 prefill 后静态选择。PyramidKV 的层级自适应策略也可以应用于 H2O 的框架中。
- **与 [[Scissorhands]] 的关系**：两者都是 token eviction 方法，Scissorhands 强调 importance 的时间持续性，PyramidKV 强调层级间的差异性，两个视角互补。
- **与 [[KV Cache Management]] 的关系**：PyramidKV 是 eviction-based 方法中首个系统考虑层级差异的工作，为后续层级感知 cache 管理方法提供了重要启发。
- **与 [[Attention Sparsity]] 的关系**：PyramidKV 的核心观察 -- 高层注意力更稀疏 -- 是 attention sparsity 领域的一个重要实证发现。

## 引用关系

### 被引用 (本文启发的后续工作) ->
- -> [[ShadowKV]]：在 offloading 策略中也考虑了层级差异
- -> [[TriAttention]]：多维注意力中的层级感知设计
- -> 后续层级自适应 cache 管理工作

### 引用的前置工作 <-
- <- [[StreamingLLM]]：attention sink 现象和 KV cache eviction 思路
- <- [[SnapKV]]：基于 prefill 注意力的 token selection 方法
- <- [[H2O]]：Heavy Hitter Oracle，动态 token eviction
- <- [[Scissorhands]]：pivotal token 持续性假设
- <- [[GQA]]：从架构层面减少 KV cache 的前置工作
