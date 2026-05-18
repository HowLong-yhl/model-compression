---
title: KV Cache Quantization
aliases: [KV Cache 量化, KV 量化, KV Cache Compression]
tags: [model-compression, quantization, concept, kv-cache, inference, long-context, activation-quantization]
created: 2026-04-28
updated: 2026-04-28
---

**父概念**：[[KV Cache Management]]（KV cache 量化是 KV cache 管理四大子方向之一）

## 核心概念

KV Cache 是 autoregressive Transformer 推理中的核心数据结构——解码时将已计算的 Key 和 Value 向量缓存起来，避免重复计算。然而 KV cache 的大小随 **序列长度 × 批量大小** 线性增长，在长上下文和高并发场景下成为 GPU 显存的主要瓶颈。

以 LLaMA-7B 为例：FP16 下 1M context length 的 KV cache 需约 128GB 显存，远超模型权重本身（14GB）。KV cache 量化旨在将这些缓存从 FP16 降低到 2-4 bit，从而在几乎不损失质量的前提下大幅压缩内存占用、提升吞吐量。

KV cache 量化是 **activation quantization** 的一个特殊子问题，但有其独特性质：KV 一旦计算完成就是静态的（不像一般激活每次前向传播都变化），且 Key 和 Value 具有截然不同的数值分布特性。

## 为什么 KV Cache 量化成为独立研究方向

传统的 weight-only 量化（[[GPTQ]]、[[AWQ]]）和 weight-activation 量化（[[SmoothQuant]]、[[OmniQuant]]）没有专门处理 KV cache。随着长上下文需求的爆发（128K→1M→10M），KV cache 成为推理的核心瓶颈：

| 场景 | 模型权重 (FP16) | KV Cache (FP16) | KV/权重 比值 |
|------|---------------|-----------------|-------------|
| LLaMA-7B, 4K context | 14GB | 0.5GB | 3.6% |
| LLaMA-7B, 128K context | 14GB | 16GB | 114% |
| LLaMA-7B, 1M context | 14GB | 128GB | 914% |
| LLaMA-70B, 128K context | 140GB | 160GB | 114% |

当 context length 超过 ~32K 时，KV cache 就超过了模型权重的大小。此时 weight-only 量化不再是内存瓶颈的主要解决方案——必须直接量化 KV cache。

## Key 与 Value 的不对称性

KV cache 量化的核心发现（由 [[KIVI]] 和 [[KVQuant]] 独立报告）是 **Key 和 Value 的 outlier 模式根本不同**，必须沿不同方向量化：

### Key Cache：固定 Outlier Channel → Per-Channel 量化

Key cache 继承了 LLM 的 [[Emergent Outlier Features]] 特征——特定通道持续产生远大于均值的极端值。这种模式跨 token 稳定，同一 channel 的值域在不同 token 间高度一致。Per-channel 量化（沿 token 维度统计）为每个通道分配独立的 scale/zero-point，精确适配其独特分布。

**RoPE 的影响**：[[KVQuant]] 进一步发现 RoPE（Rotary Position Embedding）会破坏 Key 的 channel 稳定性——RoPE 对相邻通道对施加随位置变化的旋转，使 post-RoPE 的通道分布变得位置依赖。Pre-RoPE quantization（在 RoPE 之前量化 Key）可以回避这一问题。

### Value Cache：无固定 Outlier → Per-Token 量化

Value cache 没有固定的 outlier channel 模式，per-channel 量化对 Value 效果不佳。但由于 attention 的稀疏性（softmax 输出集中在少数 token），per-token 方向的量化误差只在低权重 token 上累积，对最终输出影响很小。因此 Value 应该 per-token 量化。

**形式化**：输出 $Y = AV$，per-token 量化 Value 的误差 $\Delta Y = A \cdot E_v$。当 $A_{:,i} \approx 0$（低权重 token），即使 $e_i$ 较大也不影响输出。这与 [[Token Eviction|eviction 方法]]利用的是同一个 [[Attention Sparsity]] 性质。

## 方法分类体系

### 一、KV Cache 量化方法（本 wiki 重点）

按技术路线可分为三个流派：

#### 流派一：Calibration-based 非均匀量化

以 [[KVQuant]]（NeurIPS 2024, UC Berkeley）为代表。核心思路：利用 calibration 数据计算最优量化参数。

- **Per-Channel Key + Per-Token Value** 量化方向
- **Pre-RoPE Key Quantization**：在 RoPE 之前量化 Key
- **Non-Uniform Quantization (NUQ)**：sensitivity-weighted k-means 聚类求最优码本
- **Dense-and-Sparse**：isolate top-1% outlier 为 FP16
- 结果：3-bit <0.1 PPL 退化，单 A100 支持 1M context

#### 流派二：Tuning-free 均匀量化

以 [[KIVI]]（ICML 2024, Rice/HKUST）为代表。追求零 calibration 的极致简洁。

- **非对称量化方向**：per-channel Key + per-token Value
- **Grouped + Residual**：group-wise 量化 + sliding window 全精度缓冲区
- **Tiled matmul**：混合精度矩阵乘法 kernel
- 结果：2-bit tuning-free，2.6× 内存压缩，4× batch size

**后续改进**：

| 方法 | 年份 | 会议 | 核心改进 | 在 KIVI 基础上 |
|------|------|------|---------|---------------|
| **GEAR** | 2024 | ICML | 低秩 + 稀疏残差补偿 | 用 SVD 低秩矩阵吸收系统性量化误差，稀疏矩阵捕捉残余 outlier |
| **RotateKV** | 2025 | IJCAI | Outlier-aware adaptive rotation | 在 per-channel 量化前用自适应旋转进一步平滑 outlier |
| **KVLinC** | 2025 | — | Hadamard rotation + linear combination | Hadamard 消除 outlier 后用线性组合码本量化 |
| **WKVQuant** | 2025 | ACL | Outlier token tracing | 追踪 outlier token（如 BOS、注意力汇聚 token）并保留全精度 |
| **ZipCache** | 2025 | — | 混合量化 + channel 重排序 | 按 channel 重要性排序，重要 channel 高精度、次要 channel 低精度 |

#### 流派二·B：Layer-Wise Mixed-Precision（层间混合精度）

以 **[[KVTuner]]**（ICML 2025, Huawei/CUHK）为代表。与 intra-layer 细粒度方法正交，专注 **inter-layer** 维度的精度分配。

- **核心洞察**：不同层对 KV 量化的敏感度差异巨大且为模型固有属性（与输入无关）
- **Key > Value**：同等比特预算下高精度 Key + 低精度 Value（如 K8V4）始终优于反向配置（K4V8），因 Key 量化误差经 softmax 指数放大
- **离线搜索 + 零在线开销**：用 MOO 自动搜索每层最优 KV 精度对，在线推理直接使用配置
- **两级剪枝加速**：intra-layer Pareto 剪枝 + inter-layer 聚类，将搜索空间从 9^L 降到 S_p^G
- 结果：Llama-3.1-8B 近无损 **3.25-bit**；对 KIVI-4 崩溃的 Qwen2.5-7B 实现 4-bit 近无损；吞吐提升 **21.25%**
- **与 KIVI 正交互补**：KIVI 解决 intra-layer 问题（哪些 token 高精度），KVTuner 解决 inter-layer 问题（哪些层什么精度）

#### 流派三：理论最优量化

以 [[TurboQuant]]（ICML 2025, Google DeepMind）为代表。从信息论出发追求蒸馏率下界。

- **Random Rotation → Beta 分布 → Lloyd-Max 最优标量量化**
- **两阶段内积保持**：MSE 量化器 (b-1 bits) + QJL 残差 (1 bit)
- **Data-oblivious**：不需要 calibration 数据
- 结果：可证明在 Shannon 下界的 ~2.7× 以内，3.5 bits/dim 近无损

**相关理论方法**：

| 方法 | 年份 | 会议 | 核心思路 |
|------|------|------|---------|
| **QJL** | 2025 | AAAI | 1-bit 随机投影（Quantized JL Transform），仅保持内积 |
| **TurboQuant** | 2025 | ICML | QJL + Lloyd-Max 两阶段，近最优蒸馏率 |

#### 流派四：层间预测 + 残差量化

以 **[[AQUA-KV]]**（2025, ISTA/HSE/Yandex）为代表。利用 transformer 残差连接导致的**层间 KV 线性依赖**，训练紧凑预测器后只量化残差。

- **核心洞察**：前一层 Keys 可解释当前层 Keys ~82% 方差（Explained Variance Ratio）；前一层 Values + 当前 Keys 可解释 Values ~72% 方差
- **架构**：Linear Predictor（162MiB）→ Residual = True KV - Predicted KV → 量化 Residual（残差方差降为原始的 1/5~1/10）
- **Pre-RoPE 预测**：在 RoPE 之前做预测效果更好（post-RoPE 需要 rotation-equivariant predictor）
- **与后端量化器无关**：可搭配 HIGGS、Quanto、KIVI、KVQuant 等任意方案量化残差
- **One-shot calibration**：256 sequences × 8192 tokens，单 GPU 1-6 小时
- **可与 pruning 组合**：+ H₂O (20% tokens) 实现 38.3× 内存节省
- 结果：2-bit Qwen2.5 7B LongBench 46.43（vs HIGGS 直接崩溃 25.97），Llama3.1 70B LongBench 52.79（vs HIGGS 52.18）

**关键区别**：与其他流派不同，AQUA-KV 不改变量化器本身，而是在量化之前通过预测将需量化的数据（残差）的方差缩小，从而让任何量化器都更有效。这是一种**元方法（meta-method）**，可叠加到任何现有量化方案之上。

### 二、KV Cache 的其他压缩方法（非量化）

KV cache 量化只是 [[KV Cache Management]] 四大子方向之一。其他方向各自拥有专门的概念页：

#### Token Eviction / Merging

不压缩数值精度，而是直接丢弃或合并不重要 token 的 KV。详见 **[[Token Eviction]]**。

代表方法：[[H2O]]（NeurIPS 2023, Heavy Hitter eviction）、[[StreamingLLM]]（ICLR 2024, [[Attention Sink]] + window）、[[Scissorhands]]（ICLR 2024, pivotal token）、[[SnapKV]]（NeurIPS 2024, observation-window voting）、[[PyramidKV]]（ACL 2025, 层自适应）、[[DuoAttention]]（ICML 2025, head-level Retrieval/Streaming 二分类）、[[Ada-KV]]（NeurIPS 2025, 首个 head-wise adaptive budget allocation，理论最优全局 Top-B）、[[VL-Cache]]（arXiv 2024, VLM modality-aware post-vision attention scoring）、[[MixKV]]（ICLR 2026, importance+diversity 联合 head-wise 自适应）、[[AirCache]]（arXiv 2025, VLM elite observation window + strength/skewness budget）、[[Attention Debiasing]]（arXiv 2026, VLM recency bias + padding sink 去偏）、[[CAOTE]]（arXiv 2025, 首个 value-aware 闭合形式 eviction score，meta-heuristic 叠加任意方法）、[[CAKE]]（ICLR 2025, preference-prioritized adaptive allocation + cascading management）、[[KVzip]]（NeurIPS 2025, query-agnostic）、[[KeyDiff]]（NeurIPS 2025, attention-free Key 多样性评分）、[[Expected Attention]]（arXiv 2025, Gaussian MGF 预测未来 attention）、[[TriAttention]]（arXiv 2026, Pre-RoPE trigonometric）。

[[KeepKV]]（AAAI 2026）将 eviction 升级为理论无损的 **merging**——Electoral Votes 机制记录合并历史，ZIP-Merging 通过对数缩放 Key 确保零输出扰动。仅 10% cache budget 仍维持质量，与任意 eviction 策略和量化方法（如 [[KIVI]]）正交互补。

[[DiffKV]]（OSDI 2025）将量化与 pruning **统一为层次化框架**——利用 Key>Value 影响力差异实施差异化精度（K8V4/K4V2），同时按 token 重要性分层（高精度/低精度/剪枝），并通过 per-head per-request 动态 budget 适应不同稀疏模式。配合 Parallel KV Compaction 系统（Unified Pages + Circular Free Page List + Bidirectional Page Table），实现 2.7×–5.7× 压缩和最高 5.4× 吞吐提升。在 thinking model（QwQ-32B）上 AIME24 仅 -0.3%，远优于 INT4/KIVI/Quest/SnapKV。

**与量化的关系**：Eviction/Merging 和量化是正交且互补的——eviction 减少 token 数量，量化降低每个 token 的比特数，两者可组合使用（如 KVzip 实验：4-bit + 70% eviction = ~54× 压缩）。

#### 架构级改进

从模型架构层面减少 KV cache 的大小。详见 **[[Grouped-Query Attention]]**。

MQA → [[GQA]] → [[DeepSeek-V2-MLA|MLA]] 的演进使 KV cache 在架构层面压缩 4-93%。GQA 已成为现代 LLM 的标准配置（LLaMA-3、Mistral、Gemma 等）。量化可在 GQA 基础上进一步压缩。

#### Low-Rank Compression + Offloading

用低秩近似压缩 KV，结合 CPU offloading。详见 **[[Low-Rank KV Compression]]** 和 **[[KV Cache Offloading]]**。

[[ShadowKV]] 发现 Pre-RoPE Key 具有低秩性（SVD top-r 可捕获主要能量），将低秩部分存 GPU、残差 offload 到 CPU，配合 landmark-based sparse retrieval 实现高吞吐推理。

## 关键 Insight 总结

1. **Key per-channel, Value per-token** 是 KV cache 量化的核心原则（[[KVQuant]]、[[KIVI]] 独立验证）
2. **Pre-RoPE Key Quantization** 可显著提升 Key 的量化精度（[[KVQuant]]），因为 RoPE 破坏了通道稳定性
3. **2-bit KV cache 是可行的**（[[KIVI]]）——attention 稀疏性为 Value 的低精度量化提供了天然保护
4. **Random rotation 为量化提供理论保证**（[[TurboQuant]]）——rotation 后坐标独立且分布已知，可应用最优标量量化
5. **Eviction 和量化正交互补**——eviction 减少 token 数，量化压缩 per-token bits，组合可实现更激进的压缩

## 研究时间线

```
2023.06  FlexGen                     ← 首次在推理框架中系统考虑 KV cache 量化
2023.06  H2O (NeurIPS 2023)          ← Heavy Hitter Oracle eviction
2023.10  StreamingLLM (ICLR 2024)    ← Attention Sink 发现，infinite context window
2023.11  Scissorhands (ICLR 2024)    ← Pivotal token eviction
2024.01  KVQuant (NeurIPS 2024)      ← 4 种技术组合，3-bit <0.1 PPL 退化，1M context
2024.02  KIVI (ICML 2024)            ← Tuning-free 2-bit，非对称量化方向
2024.03  GEAR (ICML 2024)            ← 低秩 + 稀疏残差补偿框架
2024.04  SnapKV (NeurIPS 2024)       ← Observation window voting eviction
2024.05  DeepSeek-V2 (arXiv)         ← 架构级 MLA 潜空间投影
2024.06  QJL (AAAI 2025)             ← 1-bit JL transform for KV cache
2024.06  IntactKV (ACL 2024)         ← Pivot token 全精度 KV 保护
2024.10  PyramidKV (ACL 2025)        ← 层级自适应 cache budget
2024.10  DuoAttention (ICML 2025)    ← Head 二分类 Retrieval/Streaming + optimization-based 识别，与量化正交组合支持 3.3M tokens
2024.07  Ada-KV (NeurIPS 2025)       ← 首个 head-wise adaptive budget allocation，L₁ loss upper bound 理论 + plug-and-play
2024.10  VL-Cache (arXiv 2024)       ← 首篇 VLM 专用 eviction：modality boundary + post-vision attention scoring，与量化完全正交
2025.01  MixKV (SJTU, ICLR 2026)    ← Importance+Diversity 联合 head-wise 自适应混合，VLM+LLM plug-and-play 框架
2025.07  AirCache (Alibaba 2025)     ← VLM elite observation window + strength/skewness budget，跨架构（LLaVA-OV/InternVL2/Qwen2-VL）验证
2025.01  CAKE (ICLR 2025)           ← Layer-preference P2A + cascading management，3.2% cache 保性能
2025.07  DiffKV (OSDI 2025)          ← 量化+Pruning 统一框架：K8V4/K4V2 + 层次化 token compression + Parallel KV Compaction，5.4× throughput
2024.12  ShadowKV (arXiv 2025)       ← Pre-RoPE SVD + CPU offload + sparse retrieval
2025.01  RotateKV (IJCAI 2025)       ← Outlier-aware adaptive rotation + 2-bit
2025.02  TurboQuant (ICML 2025)      ← 近最优蒸馏率，理论框架
2025.02  KVLinC                      ← Hadamard + linear combination codebook
2025.05  WKVQuant (ACL 2025)         ← Outlier token tracing
2025.05  ZipCache                    ← Mixed-precision channel reordering
2025.05  KVzip (NeurIPS 2025)        ← Query-agnostic context reconstruction
2025.06  KeyDiff (NeurIPS 2025)      ← Attention-free Key 多样性评分，block processing 友好
2025.06  Expected Attention (arXiv 2025) ← Gaussian MGF 预测未来 attention，prefill+decode 通用
2025.11  KeepKV (AAAI 2026)          ← Eviction→Merging, Electoral Votes + ZIP-Merging 零扰动
2025.07  EVICPRESS (UChicago 2025)   ← 联合 compression+eviction：per-context utility function + multi-tier 存储最优放置
2025.10  CAOTE (Qualcomm AI 2025)    ← 首个 value-aware 闭合形式 eviction score，meta-heuristic 叠加任意 attention 方法
2025.05  AQUA-KV (ISTA/HSE 2025)     ← 层间预测+残差量化：Linear Predictor 将 KV 方差降 5-10×，2-bit 近无损，与任意量化器/pruning 兼容
2026.04  TriAttention (arXiv 2026)   ← Pre-RoPE trigonometric scoring
```

## 开放问题

1. **量化 + Eviction 的最优联合策略**：两者如何最优组合？是否应先 evict 再量化，还是联合优化？[[EVICPRESS]] 首次提出 per-context 联合优化框架（通过 utility function），但仅支持 token dropping 类压缩；将 quantization 方法（KIVI 等）纳入同一框架仍为 future work
2. **GQA/MLA + 量化的交互**：GQA 减少了 KV head 数，每个 head 的量化误差影响更多 query head——GQA 下量化是否更困难？MLA 的潜空间量化是否需要特殊处理？
3. **2-bit 以下是否可行？**：KIVI 实现了 2-bit，1-bit KV cache 的理论下界在哪里？TurboQuant 的 rate-distortion 分析可以提供指导
4. **Video/Multimodal LLM 的 KV cache**：视频 LLM 的 KV cache 极大（视频帧数 × token 数），但时间维度上有冗余——跨模态 KV cache 压缩是新方向
5. **与 Speculative Decoding 的结合**：低精度 KV cache 是否可以用于 draft model 的快速验证？

## 引用关系

← 父概念：[[KV Cache Management]]（量化是 KV cache 管理四大子方向之一）
← 理论根基：[[Emergent Outlier Features]]（Key 的固定 outlier channel），[[Attention Sparsity]]（Value 量化鲁棒性的理论保证），[[Incoherence]]（rotation 的理论基础），Shannon Rate-Distortion Theory
← 技术关联：[[Rotation-based Quantization]]（TurboQuant 的旋转框架），[[Equivalent Scaling Transformation]]（SmoothQuant 的 outlier 迁移思想影响了 GEAR），[[Quantization Granularity]]（per-channel / per-token / per-group 的选择）
← 已录入来源：[[KVQuant]]（NeurIPS 2024），[[KIVI]]（ICML 2024），[[TurboQuant]]（ICML 2025），[[KVTuner]]（ICML 2025），[[KeepKV]]（AAAI 2026），[[DiffKV]]（OSDI 2025），[[AQUA-KV]]（2025）
→ 正交子方向：[[Token Eviction]]（减少 token 数），[[Low-Rank KV Compression]]（低秩近似），[[KV Cache Offloading]]（分层存储），[[Grouped-Query Attention]]（架构级 KV 压缩）
→ 与 LLM 量化方法的关系：[[QuaRot]] 和 [[SpinQuant]] 的 W4A4**KV4** 设定本质上也做了 KV cache 量化，但它们将 KV 量化视为全量化的一部分而非独立问题
→ 应用场景：长上下文推理（128K-10M），高并发 serving（batch size 扩大），边缘设备部署
