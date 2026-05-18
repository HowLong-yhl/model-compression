---
title: Token Eviction
aliases: [KV Cache Eviction, Token Pruning, KV Eviction, Token 驱逐]
tags: [model-compression, concept, kv-cache, eviction, attention-sparsity]
created: 2026-04-28
updated: 2026-04-28
sources: [H2O, SnapKV, Scissorhands, StreamingLLM, KVzip, TriAttention, PyramidKV, KeepKV, KeyDiff, Expected Attention, TOVA, CAKE, DiffKV, EVICPRESS, AQUA-KV, DuoAttention, VL-Cache, AirCache, MixKV, Attention Debiasing, CAOTE, Ada-KV]
---

## 核心概念

Token Eviction 是 KV cache 管理的一大子方向——通过识别并移除不重要 token 的 KV 对来压缩 cache，依赖 [[Attention Sparsity]]（>95% 的 attention 权重接近零）的观察。与 [[KV Cache Quantization]]（降低每个 token 的比特数）正交互补。

## 三轴分类体系

Token eviction 方法可沿三个正交维度分类，形成类似量化领域 [[A Comprehensive Evaluation on Quantization Techniques for LLMs|两步分解]] 的统一框架：

### 轴一：Query 依赖性

| 类型 | 描述 | 代表方法 | 优势 | 劣势 |
|------|------|---------|------|------|
| **Query-Aware** | 基于当前 query 的 attention 信息评估 KV 重要性 | [[H2O]], [[SnapKV]], [[Scissorhands]] | 对当前 query 最优 | 多 query 复用时退化；每个 query 需重新评估 |
| **Query-Agnostic** | 不依赖具体 query，评估 KV 对"通用上下文信息"的贡献 | [[KVzip]], [[TriAttention]], [[KeyDiff]] | 压缩一次、跨 query 复用 | 对特定 query 可能不如 query-aware 精准 |

[[KVzip]] 系统化地揭示了这一维度——其实验证明 SnapKV-reuse（将 query-aware 的压缩结果复用于新 query）在 60% cache 下精度仅 40%，而 KVzip 的 query-agnostic 方案在 30% cache 下保持 95%。

### 轴二：Eviction 时机

| 类型                  | 描述                     | 代表方法                                                                  |
| ------------------- | ---------------------- | --------------------------------------------------------------------- |
| **Prefill-time**    | 在 prompt 处理完成后一次性压缩    | [[SnapKV]], [[KVzip]]                                                 |
| **Generation-time** | 在解码过程中逐步 eviction      | [[H2O]], [[Scissorhands]],[[TriAttention]]                            |
| **Offline**         | 基于离线 calibration 的静态策略 | TriAttention 的 Q/K center calibration, KVzip 的 context-independent 模式 |

在实际部署中（长 prompt + 短生成），prefill-time 压缩更有价值，因为 prompt KV 占绝大部分内存。

### 轴三：Budget 分配

| 类型 | 描述 | 代表方法 |
|------|------|---------|
| **Fixed (head-uniform)** | 所有 layer 和 head 使用相同 cache budget | [[H2O]], [[SnapKV]] |
| **Layer-adaptive** | 不同层分配不同 budget | [[PyramidKV]]（固定 pyramid 形）, [[CAKE]]（数据驱动 preference score） |
| **Head-adaptive** | 不同 head 分配不同 budget | [[Ada-KV]]（首个 head-adaptive，理论最优全局 Top-B 分配）, [[KVzip]]（non-uniform head budget） |

## 方法演进

```
2023.06  H2O (NeurIPS 2023)           ← 奠基：Heavy Hitter + greedy eviction
2023.10  StreamingLLM (ICLR 2024)     ← Attention Sink 发现，sink + window
2023.11  Scissorhands (ICLR 2024)     ← Pivotal token persistence hypothesis
2024.01  TOVA (arXiv 2024)            ← 无假设纯 attention-score eviction，MSRNN 理论框架
2024.04  SnapKV (NeurIPS 2024)        ← Prefill-time compression，observation-window voting
2024.06  PyramidKV (ACL 2025)         ← Layer-adaptive budget (pyramid shape)
2024.10  VL-Cache (arXiv 2024)        ← 首篇 VLM 专用 eviction：modality boundary 发现 + post-vision attention scoring，10% cache 98%+ 精度
2024.10  DuoAttention (ICML 2025)    ← Head 二分类（Retrieval vs Streaming），optimization-based 识别，MHA 2.55× / GQA 1.67× 内存压缩
2024.07  Ada-KV (NeurIPS 2025)       ← 首个 head-wise adaptive budget allocation：L₁ loss upper bound 理论 + 全局 Top-B 分配，plug-and-play 增强 SnapKV/Pyramid
2025.01  CAKE (ICLR 2025)            ← Layer-preference adaptive allocation + cascading management
2025.07  DiffKV (OSDI 2025)          ← 量化+Pruning 统一：K8V4/K4V2 差异化 + per-head dynamic sparsity + Parallel KV Compaction
2025.07  EVICPRESS (UChicago 2025)   ← 联合 compression+eviction：per-context 自适应方法/比率 + multi-tier 放置 + utility function
2025.01  MixKV (SJTU, ICLR 2026)    ← VLM+LLM：importance+diversity 联合，head-wise 冗余度自适应混合，plug-and-play 框架，极端压缩+5.1%
2025.07  AirCache (Alibaba 2025)     ← VLM：elite observation window + strength/skewness budget，跨架构验证，1% cache 比 SnapKV +6.5%
2025.05  AQUA-KV (ISTA/HSE 2025)     ← 层间预测+残差量化，与 H₂O 组合实现 38.3× 压缩
2025.10  CAOTE (Qualcomm AI 2025)    ← 首个 value-aware 闭合形式 eviction score：eviction error ≡ CAOTE score（Theorem 3.2），meta-heuristic 叠加任意 attention 方法，H2O+CAOTE 提升 >100%
2025.05  KVzip (NeurIPS 2025)         ← Query-agnostic，context reconstruction
2025.06  KeyDiff (NeurIPS 2025)       ← Attention-free，Key 多样性评分，block processing 友好
2025.06  Expected Attention (arXiv 2025) ← Gaussian MGF 预测未来 attention，prefill+decode 通用
2025.11  KeepKV (AAAI 2026)           ← Eviction→Merging 升级，Electoral Votes + ZIP-Merging 零扰动
2026.01  Attention Debiasing (arXiv 2026) ← Recency bias + padding attention sink 去偏，plug-and-play 增强 VLM pruning
2026.04  TriAttention (arXiv 2026)    ← Pre-RoPE trigonometric scoring，长推理
```

## 与量化的正交互补

Eviction 减少 token 数量 N，量化降低每个 token 的比特数 b。总 KV cache 大小 ∝ N × b，因此两者可组合：

$$\text{压缩比} = \frac{N_{\text{original}} \times b_{\text{original}}}{N_{\text{evicted}} \times b_{\text{quantized}}}$$

例如 [[KVzip]] 实验：16-bit KV → 4-bit 量化 + 70% eviction = **~54× 压缩**（16.3GB → 1.2GB），精度近无损。[[AQUA-KV]] + H₂O (20% tokens) 在 2-bit 残差量化下实现 **38.3× 内存节省**（Llama3.1 8B LongBench 40.88 vs H₂O only 41.42），进一步验证了 eviction 与量化组合的巨大潜力。

## 核心挑战

1. **不可恢复性**：所有 eviction 方法都是单向的——被驱逐的 KV 永久丢失。多轮对话中可能累积信息损失（[[ShadowKV]] 通过 offload 而非 eviction 回避了这个问题）
2. **评估指标的局限性**：PPL 和 accuracy 可能无法捕捉 eviction 导致的细微信息丢失（如特定事实的遗忘）
3. **长程依赖的保护**：在需要长程记忆的任务中（如 NIAH、DFS 递归），eviction 需要特别小心——[[TriAttention]] 的三角级数方法在这方面表现更好

## Attention-Free Eviction：Key 多样性

[[KeyDiff]]（NeurIPS 2025, Qualcomm AI Research）提出完全**不依赖 attention score** 的 eviction 评分，专为资源受限的 block prompt processing 设计：

- **核心洞察**：Key 向量间的 pairwise cosine similarity 与 attention score 呈负相关——几何上独特的 Key 获得高注意力
- **方法**：保留与 anchor vector（Key 均值）cosine similarity 最低的 N 个 KV 对（O(n) 复杂度）
- **理论保证**：KEYDIFF 求解最大化 Key 多样性的子集选择问题；两定理证明高 attention 的 Key 必然具有低 Key-Mean 相似度
- **FlashAttention 兼容**：不需 materialize attention matrix → TTFT 比 attention-based 方法降低 30%
- **Block processing 鲁棒性**：Key 几何关系不依赖 query → 不受 block 分割影响，在极限压缩（2K cache ~80% 压缩）下远优于所有 attention-based 方法
- 结果：Llama-3.1-8B LongBench 8K cache（23% 压缩）仅 **-0.17** 精度损失

与 [[KeepKV]] 的互补：KeyDiff 改进 **scoring**（选哪些 token），KeepKV 改进 **action**（eviction→merging）。

## 分布预测 Eviction：Expected Attention

[[Expected Attention]]（arXiv 2025, Sapienza/NVIDIA）通过**预测未来 query 的期望 attention** 实现有理论根据的 eviction，同时适用于 prefill 和 decode 两阶段：

- **核心洞察**：LLM hidden state 近似高斯分布 → query 也是高斯 → 利用 MGF 闭合形式计算期望 unnormalized attention score
- **重要性度量**：‖Δ̂h_i‖ = (â_i + ε) · ‖W_o v_i‖，同时考虑期望 attention weight 和 value 影响力
- **位置平均化**：对未来 T=512 位置的 RoPE 取均值，创建位置无关的 query 分布
- **Head-adaptive 压缩**：不同 head 保留不同数量的 KV 对
- **双阶段通用**：不依赖过去的 attention score，可在 prefill（prompt 压缩）和 decode（生成压缩）中使用
- 结果：Qwen3-8B Ruler 50% 压缩仅 **-0.6**（94.7 vs 95.3）；MATH-500 推理 12× 压缩下全面领先

**与 KeyDiff 的关系**：两者都是 FlashAttention-compatible 的双阶段方法，但理论基础不同——KeyDiff 用 Key 几何多样性（静态属性），EA 用未来 query 的分布预测（动态预估）。EA 在 Qwen3/Gemma3 上优于 KeyDiff，但 Llama3.1 极高压缩下 KeyDiff 更鲁棒。

## Layer-Adaptive Eviction：CAKE

[[CAKE]]（ICLR 2025, SJTU/Ant Group）首次系统性地建模 **layer 间注意力异质性**，提出 Preference-Prioritized Adaptive Allocation (P2A)：

- **核心洞察**：不同 layer 的 attention 空间分散度（entropy H）和时间偏移量（variance V）差异巨大 → 相同全局 budget 下，不同 layer 应获得不等的 cache 份额
- **Preference Score**：P = H^τ₁ · V^τ₂，统一量化每层的 cache 需求程度
- **Cascading Management**：prefilling 过程分 L 个 stage 逐层处理，每加入新 layer 即全局重新分配并 evict → 峰值内存始终受控
- **Attention-Shift Tolerant Eviction**：I[n] = Mean(A[:,n]) + γ·Var(A[:,n])，加法组合兼顾持续重要性和注意力波动性
- 结果：**3.2% KV cache** 保持性能，128K context **10× 加速**，LongBench/NeedleBench 全面优于 StreamingLLM/H2O/TOVA/SnapKV/PyramidKV

**与 PyramidKV 的关系**：PyramidKV 用固定 pyramid 形状做 layer-adaptive allocation，CAKE 则用数据驱动的 preference score 实现真正自适应——实验表明 P2A 一致优于 pyramid/uniform/random 策略。

## Eviction → Merging 升级

[[KeepKV]]（AAAI 2026, Peking University/ByteDance/Tsinghua）将 eviction 升级为**理论无损的 merging**，直接解决核心挑战 #1（不可恢复性）：

- **核心问题**：凸组合 merging 导致 **Attention Sag**——合并后 KV 对的注意力分数必然低于原始多个 KV 对之和
- **Electoral Votes 机制**：记录每个 KV 对的合并次数 p_i，注意力计算中乘以投票数补偿
- **ZIP-Merging**：通过对数缩放 Key + 注意力加权 Value，保证当前步零输出扰动（Theorem 3）
- **多步误差界**：EMA 预测注意力分数，理论误差 Θ < 2ε(1+ε)γ/(1-ε)²
- **无约束组合**：可与 H2O/SnapKV/PyramidKV 等任意 eviction 策略组合，用 KeepKV 替代丢弃步骤
- 结果：仅 **10% cache** 仍维持质量，吞吐 **2.2×**

## Value-Aware Eviction：CAOTE

[[CAOTE]]（arXiv 2025, Qualcomm AI Research）首次以闭合形式将 attention score 和 value 向量整合为统一 eviction 评分，直接优化 eviction 前后 attention output 的 MSE：

- **核心公式**：$c_j = \frac{\alpha_j}{1-\alpha_j} \|VA^\top - v_j\|_2$，其中 $VA^\top$ 是完整 attention output
- **精确等价**：证明 CAOTE score ≡ eviction error（Theorem 3.2），不是近似或 upper bound
- **Meta-heuristic**：可叠加在任何 attention-based 方法上（H2O/TOVA/SnapKV），只需将原始 score 归一化到 $\sum=1$
- **FastCAOTE**：用 value 均值替代 attention output，Spearman 相关性 ≥0.8，计算开销极低
- 结果：H2O+FastCAOTE 在 Llama 3.1-8B 2K budget 下从 **16.89→34.07**（+100%），NIAH 4K 下 **0.330→0.568**

与 [[KeyDiff]] 的关系：同为 Qualcomm AI Research，但路线不同——KeyDiff 完全放弃 attention score，CAOTE 在 attention 基础上整合 value 信息。与 [[KeepKV]] 正交——CAOTE 改进 scoring，KeepKV 改进 action。

## 引用关系

← 理论基础：[[Attention Sparsity]]（eviction 的可行性保证），[[Attention Sink]]（必须保留 sink token）
← 已录入方法：[[H2O]]、[[SnapKV]]、[[Scissorhands]]、[[StreamingLLM]]、[[KVzip]]、[[TriAttention]]、[[PyramidKV]]、[[KeepKV]]、[[KeyDiff]]、[[Expected Attention]]、[[CAKE]]、[[DiffKV]]、[[EVICPRESS]]、[[AQUA-KV]]、[[DuoAttention]]、[[VL-Cache]]、[[AirCache]]、[[MixKV]]、[[Attention Debiasing]]、[[CAOTE]]、[[Ada-KV]]
→ VLM 特化：[[VLM KV Cache]]（VLM 的 modality-aware eviction 策略，详见子目录）
→ 与量化正交：可与 [[KIVI]]、[[KVQuant]]、[[TurboQuant]]、[[AQUA-KV]] 组合（AQUA-KV + H₂O 实现 38.3× 内存压缩）
→ 父概念：[[KV Cache Management]]（eviction 是 KV cache 管理四大子方向之一）
