---
title: "DiffKV: Differentiated Memory Management for Large Language Models with Parallel KV Compaction"
aliases: [DiffKV, Differentiated KV, Parallel KV Compaction]
source_type: paper
source_url: "https://arxiv.org/abs/2505.xxxxx"
source_date: 2025-07-01
ingested: 2026-05-12
venue: OSDI 2025
authors: [Yanqi Zhang, Yuwei Hu, Runyuan Zhao, John C.S. Lui, Haibo Chen]
affiliation: Huawei / The Chinese University of Hong Kong / Shanghai Jiao Tong University
code: null
tags: [model-compression, kv-cache, quantization, eviction, mixed-precision, memory-management, serving, system, thinking-models]
---

## 核心要点

DiffKV 是一个 KV cache 压缩的**系统级框架**，核心洞察是 KV cache 存在**三层差异化（differentiation）**，应被区别对待而非统一处理：

1. **Key vs Value 差异**：Key 对 attention 的影响远大于 Value → Key 高精度、Value 低精度（K8V4 / K4V2）
2. **Token 重要性差异**：不同 token 的 attention score 差异巨大（7 个数量级）→ 分层量化（high-prec / low-prec / pruned）
3. **Per-head 动态稀疏性差异**：不同 head、不同 request 的稀疏模式不同 → per-head per-request 自适应分配

![[images/DiffKV_fig1_allocation_patterns.png]]

关键创新同时涵盖**算法**和**系统**两个层面：

- **算法层**：Differentiated KV quantization（K8V4/K4V2）+ 层次化 token compression + 序列长度自适应阈值 + per-head 动态 budget
- **系统层**：Parallel KV Compaction——通过 GPU-resident 数据结构（Unified Pages + Circular Free Page List + Bidirectional Page Table）高效管理因差异化压缩导致的不规则内存碎片

结果：2.7×–5.7× KV cache 压缩，near-lossless 精度（包括 thinking model QwQ-32B 上 AIME24 仅 -0.3%），throughput 1.9×–5.4×（vs vLLM），内存管理开销 <0.9%。

## 动机：三层差异化观察

### Level 1：Key > Value 的影响力

![[images/DiffKV_fig2_score_vs_norm.png]]

将 attention 输出重写为单位向量的加权和：

```
Attn(Q,K,V)ᵢ = Σⱼ [softmax(QK^T/√d)ᵢⱼ · |vⱼ|] · (vⱼ/|vⱼ|)
                 ─────────────── coefficient ──────────  ─ unit vector ─
```

每个 token 对输出的贡献取决于 coefficient = attention_score × ‖v‖。实验发现：

- **Attention score**（由 Key 决定）跨越 **7 个数量级**（10⁻⁶ ~ 10⁰）
- **Value norm** 仅跨越 **~2 个数量级**

因此 Key 的精度对最终输出影响远大于 Value。验证：K8V4 ≈ FP16 精度，而镜像配置 K4V8 在 Qwen2.5-7B 上精度降至 ~0%。

### Level 2：Token 间重要性的巨大差异

![[images/DiffKV_fig4_5_critical_tokens.png]]

- 保留 95% attention score 所需的 "critical tokens" 数量在 layer 间差异 2× 以上
- 在同一 layer 内，不同 KV head 的 critical token 数也差异显著
- 同一 KV head 在不同 request 间的 critical token 数变化也很大（高标准差）

→ 统一的 pruning ratio（如 H2O、SnapKV）或固定 pattern（如 PyramidKV）无法适应这种动态性。

### Level 3：Per-head 动态稀疏性

每个 attention head 的稀疏模式不仅在 layer 间不同，甚至在同一 head 的不同 request 间也会变化。这是 DiffKV 的 per-head per-request 动态分配的核心动机。

## 方法：KV Compression Policy

### 层次化压缩策略

DiffKV 将 token 分为三级：

| 级别 | 量化精度 | 判定条件 | 内存占比 |
|------|---------|---------|---------|
| **重要** | 高精度（K8V4） | score > α_h / N | 最多 |
| **中等** | 低精度（K4V2） | α_l/N ≤ score ≤ α_h/N | 中等 |
| **不重要** | 剪枝（丢弃） | score < α_l / N | 0 |

其中 N 为序列长度，α_h 和 α_l 为离线 calibration 确定的阈值。

**序列长度自适应**：阈值除以 N → 长序列中"平均水平"自然降低 → 更多 token 被压缩/剪枝 → 自动实现"长序列多压缩、短序列多保留"。

### Prompt Phase

1. 计算所有 token 的 significance = 该 token 被后续 N-i 个 token 的平均 attention score
2. 最近 W=64 个 token 强制保留高精度
3. 其余 token 按 significance 与阈值比较，分配到三级

### Generation Phase（Algorithm 1）

每生成一个新 token：
1. 新 token 默认存入高精度 cache KV_h
2. 选出 KV_h 中 significance 最低的候选 t_c
3. 若 t_c 的 score < α_l/N → 降级到低精度 KV_l，否则保留
4. 选出 KV_l 中 significance 最低的 victim t_v
5. 若 t_v 的 score < α_l/N → 剪枝（丢弃）

GQA 模式下，多个 query head 共享一个 KV head 的 score 通过 **max** 聚合。

### 参数 Calibration

- 在 MATH 训练集上 profile（高信息密度、对压缩敏感）
- α_h ∈ [1, 5]（整数范围）
- α_l ∈ [0.02, 0.10]
- 选取在 calibration 集上达到 FP16 精度的最优值
- 所有 head 共享相同阈值（实验证明足够捕获动态稀疏性）

## 系统：Parallel KV Compaction

![[images/DiffKV_fig6_memory_management.png]]

三层差异化引入了**不规则内存使用模式**——不同 request、不同 head 的高/低精度 token 数量各异，导致 O(#requests × #heads) 复杂度的内存管理。DiffKV 通过三个 GPU-resident 数据结构实现高效 parallel compaction：

### 1. Unified Pages

GPU 内存划分为固定大小的 page，每个 page 动态配置为特定精度：

```
Page 结构 = [quantized keys | key metadata | quantized values | value metadata | scores | positions]
```

- 不同精度的 page 存放不同数量的 token（低精度 page 容纳更多 token）
- Key、Value 及 metadata 合并在单一结构中 → 提升 data locality
- 消除了因固定 page format 导致的内存浪费和 misaligned access

### 2. Circular Free Page List

- 所有 PageID 存储在 GPU 上的环形列表
- start_ptr（分配）和 end_ptr（回收）两个指针
- 分配/回收通过 **parallel prefix sum** 并行化

### 3. Bidirectional Page Table

- 每个 head 一个 page table entry
- 高精度 page ID 从左向右增长，低精度 page ID 从右向左增长
- 无需分离查找，长度由 max_seq_len / tokens_per_high_prec_page 确定

**Prompt Phase 流程**：
1. Page Allocation：保守分配（假设全部高精度）
2. Planning：各 head 独立执行压缩算法，确定实际需要的高/低精度 page 数
3. Coordination：通过 parallel prefix sum 回收多余 page

**内存管理开销**：prompt phase <0.2%, generation phase <0.9%（模型执行占 92-97%）。

## Custom Attention Kernel

DiffKV 实现了高效的混合精度 attention kernel：

- **Key processing**：布局 [F/(K_vec×K_group), N_tokens, K_group, K_vec]，确保 warp 内所有线程的内存访问连续（memory coalescing）
- **Value processing**：token 均匀分配到 thread group，各自做 sum reduction 后 tree reduction 聚合
- **On-the-fly dequantization**：量化的 KV 在 attention 计算时即时反量化，无需预先解压

延迟加速：K8V4 ~2× vs FP16，K4V2 ~4× vs FP16（与 KV cache 大小减少成正比）。

## 实验结果

### 主要精度结果（Table 1）

| Model | Benchmark | FP16 | DiffKV (Memory%) | Best Baseline |
|-------|-----------|------|-----------------|---------------|
| Llama3-8B | GSM8K | 76.3 | 75.7 (19.3%) | INT4: 72.6 |
| Llama3-8B | MATH | 28.1 | 27.9 (21.6%) | INT4: 23.9 |
| Llama3-8B | HumanEval+ | 50.0 | 48.0 (27.6%) | INT4: 45.1 |
| Qwen2.5-7B | MMLU | 75.1 | 74.7 (30.3%) | Quest: 64.7 |
| Qwen2.5-32B | MATH | 63.2 | 67.2 (25.3%) | INT4: 64.7 |
| Llama3-70B | GSM8K | 90.5 | 90.2 (21.6%) | INT4: 89.3 |

DiffKV 在 ~20-30% 内存下实现 near-lossless 精度，一致优于 INT4/KIVI/Quest/SnapKV/DuoAttn。

### Thinking Models（Table 3）

| Model | Benchmark | FP16 | DiffKV (Mem%) | INT4 | KIVI | Quest | SnapKV |
|-------|-----------|------|---------------|------|------|-------|--------|
| QwQ-32B | MATH | 90.6 | 90.6 (26.8%) | 87.3 | 80.6 | 63.3 | 65.4 |
| QwQ-32B | AIME24 | 75.5 | 75.3 (27.8%) | 64.3 | 10.0 | 23.3 | 19.2 |
| QwQ-32B | GPQA | 62.1 | 63.2 (27.6%) | 57.8 | 2.6 | 1.2 | 0.8 |
| R1-Distill-Qwen-14B | AIME24 | 67.0 | 67.0 (29.6%) | 61.6 | 10.0 | 20.2 | 17.2 |
| R1-Distill-Llama-8B | AIME24 | 51.0 | 51.0 (23.8%) | 42.3 | 6.6 | 18.2 | 15.4 |

Thinking model 上优势更为突出——KIVI/Quest/SnapKV 在 GPQA 上接近 0 分，而 DiffKV 保持 near-lossless。这表明 long CoT generation 对 KV cache 压缩更具挑战性，DiffKV 的细粒度差异化策略应对更好。

### Throughput

![[images/DiffKV_fig17_throughput.png]]

- QwQ-32B：batch size 5.9× vs vLLM，throughput **5.4×**
- Qwen2.5-32B：throughput **5.2×**
- Llama3-8B/70B：throughput 1.9×–2.0×
- 对比 Quest（仅加速 attention 而非减少内存）和 Atom/KIVI（缺乏高效 kernel），DiffKV 在所有模型上吞吐量最高

## 关键设计决策

### 为什么 K > V 精度有效？

```
Coefficient = attention_score × ‖v‖
```

- Attention score 由 Key 决定，跨越 7 个数量级 → 决定性因素
- Value norm 仅跨越 2 个数量级 → 次要因素
- Key 的 per-channel outlier + RoPE 问题使其更需要高精度
- GQA 模型尤其敏感（Qwen2.5-7B queries-per-KV ratio = 7）

### 为什么 per-head dynamic > static allocation？

- Static（H2O/SnapKV）：所有 head 相同 budget → 某些 head 过度压缩
- Per-head dynamic：根据实际稀疏性按需分配 → 在 50% pruning 下保持 full accuracy

### 为什么系统设计至关重要？

差异化压缩引入 O(#requests × #heads) 的内存管理复杂度。Llama3-8B 有数百个 head + 数百并发 request → 每步需管理数万个异构内存区域。无高效系统支持，内存管理开销会抵消压缩收益。DiffKV 的 parallel KV compaction 将此开销压缩到 <0.9%。

## 与其他方法的对比定位

| 维度 | DiffKV | KIVI | Atom/QServe | SnapKV | CAKE |
|------|--------|------|-------------|--------|------|
| **K/V 精度** | 差异化（K8V4/K4V2） | 统一（K2V2） | 统一（W4A4KV4） | N/A | N/A |
| **Token 分层** | 3 级（high/low/pruned） | 无 | 无 | 2 级（keep/evict） | 2 级（keep/evict） |
| **Budget 分配** | Per-head per-request dynamic | Uniform | Uniform | Uniform | Layer-adaptive |
| **系统优化** | Custom kernel + memory manager | HF Transformers | Custom kernel | vLLM | 无 |
| **适用场景** | Serving（高并发 + thinking model） | 通用推理 | Serving | 通用推理 | 通用推理 |
| **压缩率** | 2.7×–5.7× | 8× (bit-level) | 4× | 2×–4× | 3×–30× |
| **Throughput 提升** | 1.9×–5.4× | 受限于框架开销 | 1.5×–3.5× | ~1× | ~10× (latency) |

DiffKV 的独特贡献在于**将量化和 pruning 统一为一个层次化框架**，并配合系统级优化（parallel KV compaction）将理论压缩优势转化为实际吞吐提升。特别是对 thinking model 的长 CoT 生成场景，DiffKV 的细粒度差异化策略表现远优于纯量化或纯 pruning 方法。

## 关键公式总结

```
# Attention 分解为 coefficient × unit vector
Attn_i = Σⱼ [softmax(QK^T/√d)ᵢⱼ · |vⱼ|] · (vⱼ/|vⱼ|)

# Token significance (prompt phase)
significance(i) = (1/(N-i)) · Σₜ₌ᵢ₊₁ᴺ attention_score(t→i)

# Hierarchical compression thresholds
high-prec:  score > α_h / N
low-prec:   α_l / N ≤ score ≤ α_h / N
pruned:     score < α_l / N

# GQA score aggregation
score_KVhead = max(score across all query heads sharing this KV head)
```

## 相关链接

- 相关方法：[[KV Cache Quantization]]、[[Token Eviction]]、[[KIVI]]、[[KVQuant]]、[[CAKE]]、[[SnapKV]]、[[H2O]]
- 系统：[[QServe]]、[[Atom]]
- 综述：[[overview]]
