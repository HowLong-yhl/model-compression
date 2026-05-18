---
title: KV Cache 经验观测总结
aliases: [KV Cache Observations, KV Cache Findings, KV 经验发现]
tags: [kv-cache, empirical-findings, quantization, eviction, attention-pattern]
created: 2026-05-13
updated: 2026-05-13
sources: [KIVI, KVQuant, TurboQuant, ShadowKV, TriAttention, DuoAttention, StreamingLLM, H2O, SnapKV, CAKE, DiffKV, AQUA-KV, KVTuner, KeepKV, KeyDiff, Expected Attention, KVzip, TOVA, PyramidKV, EVICPRESS, Scissorhands]
---

# KV Cache 经验观测总结

本页汇总 KV Cache 管理研究中被多篇独立工作发现或依赖的**经验性观测事实**（empirical observations）。这些观测是所有 KV cache 压缩方法（量化、eviction、low-rank、offloading）的共同理论基础。每一条观测都标注了首次报告或关键验证的来源论文。

---

## 一、Key Cache 统计特性

### 1.1 Key 存在 Channel-Fixed Outlier

**观测**：Key cache 继承了 LLM 的 [[Emergent Outlier Features]]——特定通道持续产生远大于均值的极端值，且该模式**跨 token 稳定**：同一通道在不同 token 上的值域高度一致。outlier 通道的位置与 LLM.int8() 报告的权重 outlier 一致。

**来源**：[[KIVI]]（ICML 2024）、[[KVQuant]]（NeurIPS 2024）独立验证

**方法论意义**：这直接决定了 Key 应使用 **per-channel 量化**（沿 token 维度共享 scale），使每个通道的 outlier 被独立的 scale 覆盖。

### 1.2 Pre-RoPE Key 具有通道稳定性

**观测**：RoPE 施加前的 Key 向量具有稳定的 per-channel 分布——同一通道在不同 position 上保持一致的值域。RoPE 的位置相关旋转会**破坏**这种稳定性，使 post-RoPE 通道分布变为 position-dependent。

**来源**：[[KVQuant]]（NeurIPS 2024）

**方法论意义**：Pre-RoPE 空间是量化、低秩压缩、重要性评分的优选工作空间。多篇工作（KVQuant、ShadowKV、TriAttention、AQUA-KV）均在 Pre-RoPE 空间操作。

### 1.3 Pre-RoPE Key 具有极强低秩性

**观测**：Pre-RoPE Key cache 可通过 truncated SVD 实现 6× 以上压缩几乎无损——前 25% 奇异值捕获 >99% Frobenius 范数。Post-RoPE Key 有效秩显著更高。

**来源**：[[ShadowKV]]（arXiv 2025）

**补充观测**：低秩子空间是 data-dependent 的——不同序列有不同的主方向，不能用静态权重分解替代，需要 online SVD。

### 1.4 Pre-RoPE Q/K 方向高度集中

**观测**：Pre-RoPE 空间中 Q 和 K 向量**非**随机分布，而是高度集中在固定的非零方向中心周围。Mean Resultant Length (R) 约 90% 的 attention heads R > 0.95，表明极端方向集中性。该性质跨 position、跨 context 稳定，且跨任务域（编码数据校准的中心可直接用于推理任务）。

**来源**：[[TriAttention]]（arXiv 2026）

**方法论意义**：这使得 attention logit 可以分解为三角级数，从而实现 query-agnostic importance prediction。

### 1.5 Attention Score 跨越 7 个数量级

**观测**：attention score（由 Key 决定）跨越约 7 个数量级（10⁻⁶ ~ 10⁰），而 Value 范数仅跨越约 2 个数量级。这量化了 Key > Value 的动态范围不对称性。

**来源**：[[DiffKV]]（OSDI 2025）

### 1.6 随机旋转后坐标服从已知 Beta 分布

**观测**：对 R^d 中单位向量施加随机正交旋转后，每个坐标的平方服从 Beta(1/2, (d-1)/2)。当 d 较大（如 LLM 中 per-head d=128）时近似 N(0, 1/d)，坐标间近似独立同分布。

**来源**：[[TurboQuant]]（ICML 2025）

**方法论意义**：为旋转后均匀量化提供理论最优性保证——旋转消除方向异质性后可安全使用 per-tensor 量化。

---

## 二、Value Cache 统计特性

### 2.1 Value 无固定 Outlier 通道

**观测**：Value cache **不具有**固定 outlier 通道。其通道分布随 token 变化，无稳定 outlier 模式。Per-channel 量化在 Value 上效果显著劣于 per-token。

**来源**：[[KIVI]]（ICML 2024）、[[KVQuant]]（NeurIPS 2024）

### 2.2 Value 有效秩高，不适合低秩压缩

**观测**：Value cache 有效秩显著高于 Key cache，不适合 SVD-based 低秩压缩。

**来源**：[[ShadowKV]]（arXiv 2025）

### 2.3 KV 值近似正态分布

**观测**：KV cache 中的值近似正态分布（集中在均值附近），而非在 min/max 之间均匀分布。uniform 量化次优；non-uniform 量化（k-means codebook）更好地拟合实际分布。

**来源**：[[KVQuant]]（NeurIPS 2024）

---

## 三、Key-Value 不对称性

### 3.1 最优量化方向不同

**观测**：Key 应 per-channel（沿 token 维共享 scale），Value 应 per-token（沿 channel 维共享 scale）。最佳组合（per-channel Key + per-token Value）在 2-bit LLaMA-2-7B 上比最差组合（per-token Key + per-channel Value）低 2.58 PPL。

**来源**：[[KIVI]]（ICML 2024）、[[KVQuant]]（NeurIPS 2024）独立验证

### 3.2 Key 量化误差被 softmax 指数放大

**观测**：Key 量化误差进入 softmax 的指数（exp(qK^T/√d)），导致 attention 分布的指数级偏移。Value 量化误差仅线性传播（o = a·V）。这是 Key 需要更高精度的根本原因。

**来源**：[[KVTuner]]（ICML 2025）、[[DiffKV]]（OSDI 2025）

### 3.3 相同 bit budget 下 K>V 始终优于 K<V

**观测**：在相同总 bit budget 下，分配更多 bit 给 Key（如 K8V4）始终优于反向分配（K4V8）。Qwen2.5-7B 上，K4V8 给出 PPL=220.83，而 K8V4 给出 PPL=9.39（同为 6-bit equivalent）。

**来源**：[[KVTuner]]（ICML 2025）

### 3.4 Key 跨层可预测性强于 Value

**观测**：前一层 Key 可解释当前层 Key ~82% 方差（EVR ~0.82）；前一层 Value + 当前层 Key 可解释当前层 Value ~72% 方差（EVR ~0.72）。Key 比 Value 更易预测。temporal dependency（前一 token）几乎无帮助（Key EVR ~0.42, Value EVR ~0.17）。

**来源**：[[AQUA-KV]]（2025）

---

## 四、Attention 稀疏性与分布

### 4.1 推理时 Attention >95% 稀疏

**观测**：即使是密集训练的 LLM，推理时 >95% 的 attention weight 接近零。仅约 5% 的 KV pairs 参与有效 attention 计算。

**来源**：[[H2O]]（NeurIPS 2023）

### 4.2 Heavy Hitter 服从幂律分布

**观测**：按累积 attention score 排序 token，分布服从幂律——极少数 "Heavy Hitter" token 持续获得不成比例的高 attention，其余贡献趋近于零。

**来源**：[[H2O]]（NeurIPS 2023）

### 4.3 稀疏性随序列长度增强

**观测**：attention 稀疏性随序列长度增加而增强——更长的 context 更稀疏。

**来源**：[[H2O]]（NeurIPS 2023）

### 4.4 稀疏性为 Value 量化提供天然保护

**观测**：per-token Value 量化误差仅累积在低权重 token 上（A_{:,i} ≈ 0）。由于 attention 稀疏性，即使这些 token 的量化误差很大也不影响最终输出 Y = A·V。

**来源**：[[KIVI]]（ICML 2024）

---

## 五、Attention Sink 现象

### 5.1 初始 Token 恒定获得高 Attention

**观测**：初始 token（尤其是 BOS）**无论语义内容如何**，持续获得不成比例的高 attention score。将初始 token 替换为任意文本不消除 sink 效应——这是 position-driven，非 content-driven。移除 sink token 导致 softmax 分布坍缩和严重输出退化。

**来源**：[[StreamingLLM]]（ICLR 2024）

### 5.2 Sink 跨层跨头一致

**观测**：attention sink 模式在所有层、所有 head 中一致出现。

**来源**：[[StreamingLLM]]（ICLR 2024）

### 5.3 4 个 Sink Token 即可稳定

**观测**：仅保留 4 个初始 token 作为 attention sink 即达近最优稳定性。1 个 sink 已显著改善；超过 4 个收益递减。

**来源**：[[StreamingLLM]]（ICLR 2024）

### 5.4 Sink 的几何解释

**观测**：sink token 的 Key 向量是 Key 空间中的几何 outlier——与其他 Key 的 pairwise cosine similarity 极低。几何独特性（geometric distinctiveness）与高 attention 负相关。

**来源**：[[KeyDiff]]（NeurIPS 2025）

---

## 六、Attention 模式的稳定性与可预测性

### 6.1 Observation Window 预测后续 Attention

**观测**：prompt 最后若干 token 的 attention 模式（observation window，~32 token）可以稳定预测整个 generation 阶段哪些 token 重要。overlap rate 跨多个模型超过 90%。

**来源**：[[SnapKV]]（NeurIPS 2024）

### 6.2 重要性持续性假说

**观测**：如果一个 token 在步骤 t 处于 top-5% attention（pivotal），它在 t+1, t+2, ... 继续为 pivotal 的概率为 93-97%。该 persistence 跨层跨头普遍成立。

**来源**：[[Scissorhands]]（ICLR 2024）

### 6.3 Local Greedy 近似 Global Optimal

**观测**：仅使用已观察到的 token 的 attention score（local H2）判断重要性，结果与使用完整序列信息（global H2）几乎相同。这验证了 online eviction 的可行性。

**来源**：[[H2O]]（NeurIPS 2023）

### 6.4 三角级数可预测 Attention 重要性

**观测**：当 Q/K 方向集中时，attention logit 简化为关于 Q-K 位置距离的三角级数。三角级数预测的重要性与实际 attention pattern 的 Pearson 相关系数 r = 0.72。

**来源**：[[TriAttention]]（arXiv 2026）

### 6.5 Post-RoPE Observation Window 有限有效范围

**观测**：扩大 observation window 超过 ~25 token 不再提供额外收益，因为 RoPE 使 query 持续旋转，限制了 post-RoPE attention pattern 的时间有效性。

**来源**：[[TriAttention]]（arXiv 2026）

### 6.6 LLM Hidden State 近似高斯

**观测**：LLM 中间层 hidden state 近似服从高斯分布，跨多种架构（Llama, Qwen, Gemma）验证。由于 query 是 hidden state 的线性变换，query 继承高斯性质。这使得以闭合形式预测期望 attention 成为可能。

**来源**：[[Expected Attention]]（arXiv 2025）

---

## 七、层间异质性

### 7.1 金字塔信息漏斗

**观测**：低层 attention 更分散（均匀），需要更多 KV pair 表示；高层 attention 更集中（稀疏），需要更少 KV pair。这形成天然"金字塔"模式。

**来源**：[[PyramidKV]]（ACL 2025）

### 7.2 层间量化敏感度高度异质

**观测**：不同层对 KV cache 量化的敏感度差异巨大。该敏感度是 model-intrinsic 属性（独立于输入）。不同 prompt 的层间敏感度排序保持一致。

**来源**：[[KVTuner]]（ICML 2025）

### 7.3 空间分散度与时间偏移跨层不同

**观测**：不同层表现出截然不同的 (a) 空间 attention 分散度（熵 H——attention 分布的均匀程度）和 (b) 时间 attention 偏移量（方差 V——attention 热点随时间变化的程度）。两个指标共同刻画每层的 cache 需求。

**来源**：[[CAKE]]（ICLR 2025）

### 7.4 关键 Token 数量跨层差异 2x+

**观测**：解释 95% attention score 所需的"关键 token"数量在同一模型的不同层间差异超过 2 倍。

**来源**：[[DiffKV]]（OSDI 2025）

### 7.5 层间 KV 线性依赖（残差连接效应）

**观测**：由于 transformer 残差连接，相邻层的 KV cache 存在强线性可预测性。前一层 Key 解释当前层 Key ~82% 方差；组合前一层 Value + 当前层 Key 解释当前层 Value ~72% 方差。Pre-RoPE 空间中线性预测比 Post-RoPE 更有效。

**来源**：[[AQUA-KV]]（2025）

### 7.6 层间敏感度跨模型差异巨大

**观测**：不同 LLM 有截然不同的敏感度 profile：LLaMA-3.1-8B 可实现近无损 3.25-bit；Qwen2.5-7B 在 uniform 4-bit 下即坍缩（KV4 准确率 0.78%）。Qwen 模型有更多非稀疏 retrieval head，使 Key 量化灾难性。

**来源**：[[KVTuner]]（ICML 2025）

---

## 八、头间异质性

### 8.1 Retrieval Heads vs Streaming Heads 二分类

**观测**：LLM attention head 可分为两类：(a) Retrieval Heads——需要完整 KV cache 做长程 context 检索，(b) Streaming Heads——仅 attend to sink + recent token。MHA 模型（LLaMA-2-7B）中仅 ~25% 为 retrieval heads；GQA 模型（LLaMA-3-8B）中 ~50% KV head 为 retrieval。

**来源**：[[DuoAttention]]（ICML 2025）

### 8.2 GQA 模型 Retrieval 比例更高

**观测**：GQA 模型 retrieval head 比例（~50%）高于 MHA 模型（~25%），因为每个 GQA KV head 服务多个 query head，承载更多功能负载。

**来源**：[[DuoAttention]]（ICML 2025）

### 8.3 修改 Streaming Head 几乎无影响

**观测**：将 streaming head 限制为 sink+recent token 对 passkey retrieval accuracy 几乎无变化。修改 retrieval head 则导致性能坍缩。

**来源**：[[DuoAttention]]（ICML 2025）

### 8.4 仅稀疏集中的 Head 对低精度 KV 鲁棒

**观测**：（KVTuner Lemma 1）只有 attention 分布稀疏且集中的 head 对低精度 KV 量化鲁棒。非稀疏 retrieval head 对 Key 量化极端敏感。

**来源**：[[KVTuner]]（ICML 2025）

### 8.5 Per-Head 动态稀疏性随请求变化

**观测**：每个 attention head 的稀疏模式不仅跨层不同，也在不同请求间变化（关键 token 数量的标准差较大），需要 per-request adaptive strategy。

**来源**：[[DiffKV]]（OSDI 2025）

### 8.6 Attention Score 不完全反映 Head 重要性

**观测**：attention score 不能完全反映 head 重要性，因为：(1) 忽略 Value state 的作用，(2) 忽略端到端输出影响，(3) 忽略层间 scale 差异。optimization-based 识别（最小化端到端输出偏差）更准确。

**来源**：[[DuoAttention]]（ICML 2025）

---

## 九、时间动态

### 9.1 KV Cache 主导性随序列长度增长

**观测**：context length 超过 ~32K 时，KV cache 超过模型权重大小。1M context 下 LLaMA-7B KV cache 达 128GB（权重仅 14GB），占比 914%。

**来源**：[[KV Cache Management]]

### 9.2 量化吞吐增益随长度增大

**观测**：KV cache 量化在更长 context 下提供更大吞吐提升（KIVI: 128K 时 3.47× vs 4K 时 2.35×），因为 KV cache 在总内存中的占比随序列长度增加。

**来源**：[[KIVI]]（ICML 2024）

### 9.3 Eviction 错误不可逆且可能累积

**观测**：所有 eviction 方法都是单向的——被驱逐的 KV pair 永久丢失。多轮对话中可能造成累积信息损失。当时被判定不重要的 token 在后续轮次中可能变得重要。

**来源**：[[KeepKV]]（AAAI 2026）

### 9.4 相邻解码步选择高度重叠的 KV 块

**观测**：相邻解码步的 KV selection 有极强时间局部性——相邻步通常选择高度重叠的 chunk，使 GPU 侧缓存可减少约 60% PCIe 数据传输。

**来源**：[[ShadowKV]]（arXiv 2025）

### 9.5 Post-RoPE Key 具有空间局部性

**观测**：相邻 token 的 post-RoPE Key 向量有极高 cosine similarity，允许 chunk-level 近似（用 chunk 均值作 landmark）。

**来源**：[[ShadowKV]]（arXiv 2025）

### 9.6 量化误差沿层和序列两维累积

**观测**：KV cache 量化误差在两个维度累积：(a) 层维度（早层误差通过 hidden state 传播到后层），(b) 序列维度（带误差的前步输出成为后步输入）。微小 per-token per-layer 误差可级联为 token flipping（如 "-" 变 "+"）。

**来源**：[[KVTuner]]（ICML 2025）

### 9.7 长 CoT 生成对 KV 压缩更敏感

**观测**：thinking model（QwQ-32B）生成长 chain-of-thought 时对 KV cache 压缩远比标准模型敏感。KIVI/Quest/SnapKV 在 thinking model 上 GPQA 准确率接近 0%，而 DiffKV 的细粒度差异化策略保持近无损。

**来源**：[[DiffKV]]（OSDI 2025）

---

## 十、Token 保留模式

### 10.1 TOVA 的 Token Survival 分析

**观测**：(a) 仅 73-76% 被保留 token 是 recent（recency 不是全部）；(b) 第一个 token 始终被保留至序列末尾；(c) 最持久保留的 token 类型包括所有格（'s）、引号、句末标点、专有名词和换行符。

**来源**：[[TOVA]]（arXiv 2024）

### 10.2 几何独特 Key 获得高 Attention

**观测**：Key 向量的 pairwise cosine similarity 与 attention score 负相关。几何独特的 Key（与其他 Key 相似度低的）倾向于获得高 attention。这为 attention-free 的重要性评估提供了替代路径。

**来源**：[[KeyDiff]]（NeurIPS 2025）

### 10.3 Key Diversity 对 Block 处理鲁棒

**观测**：Key 几何关系不依赖任何 query，使基于 Key diversity 的 eviction score 天然不受 block partitioning 影响。在极端压缩（2K cache，~80% 压缩）时，KeyDiff（44.33）远超所有 attention-based 方法（H2O 16.89, TOVA 37.52, SnapKV 39.60）。

**来源**：[[KeyDiff]]（NeurIPS 2025）

---

## 十一、Context 级异质性

### 11.1 压缩敏感度跨 Context 差异巨大

**观测**：不同 context（文档/prompt）对 KV cache 压缩的敏感度截然不同。如 Wikipedia 文章 median sensitivity 0.681（高度敏感），而故事叙事仅 0.340（因对话冗余）。没有单一压缩方法/比率对所有 context 最优。

**来源**：[[EVICPRESS]]（2025）

### 11.2 最优压缩方法是 Context-Dependent 的

**观测**：测试 context 中 40.3% 最适合 kvzip，40.0% 最适合 knorm，19.7% 最适合 snapkv。三组的长度分布高度重叠——context length 本身无法预测最优方法。

**来源**：[[EVICPRESS]]（2025）

---

## 十二、Merging 与信息保存

### 12.1 凸组合 Merging 必导致 "Attention Sag"

**观测**：（定理）通过凸组合（加权平均）merge KV pair 必然导致 merged pair 的 attention score **低于**原始 pair attention score 之和。这对任何凸 merging scheme 不可避免。

**来源**：[[KeepKV]]（AAAI 2026）

### 12.2 ZIP-Merging 实现当前步零扰动

**观测**：（定理 3）ZIP-Merging 通过对数 Key 缩放 + attention 加权 Value 平均，在当前解码步实现数学上精确的零输出扰动。

**来源**：[[KeepKV]]（AAAI 2026）

---

## 十三、Cross-Cutting 发现

### 13.1 Eviction 与量化正交可乘法组合

**观测**：compression_ratio = (N_original × b_original) / (N_evicted × b_quantized)。实践：4-bit 量化 + 70% eviction = ~54× 压缩近无损。AQUA-KV + H2O（20% tokens）在 2-bit 达 38.3× 内存节省。

**来源**：[[KVzip]]（NeurIPS 2025）、[[AQUA-KV]]（2025）

### 13.2 残差预测将量化目标方差降低 5-10×

**观测**：通过前一层预测 KV 值并仅量化残差，量化目标方差降为原始的 1/5 ~ 1/10。这使任何下游 quantizer 显著更有效（meta-method，改善所有 quantizer）。

**来源**：[[AQUA-KV]]（2025）

### 13.3 Query-Aware Eviction 在多 Query 复用时严重退化

**观测**：query-aware 压缩结果（如 SnapKV）复用于新 query 时，60% cache retention 仅达 ~40% accuracy，而 query-agnostic 方法（KVzip）在 30% cache 下维持 ~95% accuracy。

**来源**：[[KVzip]]（NeurIPS 2025）

### 13.4 Reconstruction Attention 比 Normal Attention 更稀疏

**观测**：当 LLM 被 prompt 重复/重建其 context 时，reconstruction attention 比 normal prefill attention 更稀疏，能更好区分 KV 重要性。

**来源**：[[KVzip]]（NeurIPS 2025）

### 13.5 金字塔分配一致优于均匀分配

**观测**：data-driven layer-adaptive 分配（CAKE 的 P2A）一致优于 uniform、pyramid、random 分配策略。在固定模式中，pyramid 形优于 hourglass 和 inverse-pyramid。

**来源**：[[CAKE]]（ICLR 2025）、[[PyramidKV]]（ACL 2025）

### 13.6 更大模型对 KV 量化更鲁棒

**观测**：更大模型量化退化更小。LLaMA-2-70B 3-bit 仅 +0.03 PPL（vs LLaMA-7B +0.06）。2-bit: LLaMA-2-70B +0.02, LLaMA-2-7B +0.09。

**来源**：[[KIVI]]（ICML 2024）、[[KVQuant]]（NeurIPS 2024）

---

## 观测间的逻辑关系

上述观测并非孤立事实，它们之间存在因果和互补关系：

```
Key Channel Outlier (1.1)
  ├── → 决定了 per-channel 量化方向 (3.1)
  ├── → Pre-RoPE 保持稳定 (1.2) → 优选 Pre-RoPE 空间操作
  └── → 几何 outlier 解释 Attention Sink (5.4)

Attention Sparsity (4.1-4.3)
  ├── → 保护 Value 量化 (4.4)
  ├── → 使 Eviction 可行 (Token Eviction 全部方法)
  ├── → 稀疏 head 对量化鲁棒 (8.4)
  └── → 金字塔分层 (7.1)

Pre-RoPE 低秩 (1.3) + 方向集中 (1.4)
  ├── → ShadowKV 低秩压缩
  ├── → TriAttention 三角级数预测
  └── → AQUA-KV 层间预测

Key > Value 不对称 (3.1-3.4)
  ├── → K 需更高精度 (3.3)
  ├── → K 易预测 → residual quantization (3.4)
  └── → 差异化 K/V 策略 (DiffKV)

Head 异质性 (8.1-8.5)
  ├── → DuoAttention 差异化存储
  ├── → KVTuner per-head bit 分配
  └── → per-request adaptive (DiffKV)
```

---

## 方法论启示

这些观测共同指向几条设计原则：

1. **利用不对称性**：Key 和 Value 应使用不同策略（量化方向、精度分配、压缩方法）
2. **在 Pre-RoPE 空间操作**：通道稳定性、低秩性、方向集中性均在 Pre-RoPE 空间最强
3. **利用层间/头间异质性**：uniform 策略次优，adaptive per-layer/per-head 分配一致更好
4. **正交组合实现乘法压缩**：量化 × eviction × low-rank 可安全叠加（但在极端设置下需验证 interaction）
5. **Attention 稀疏性是多个方法的共同理论基础**：eviction、Value 量化容忍度、head 分类均依赖此性质
6. **持续性保证 online 策略可行**：重要性的高 persistence 使 greedy eviction 近似全局最优

---

## 相关页面

- [[KV Cache Quantization]]：量化方法综述
- [[Token Eviction]]：eviction 方法综述
- [[KV Cache Management]]：KV cache 管理总览
- [[Attention Sparsity]]：attention 稀疏性专题
- [[Attention Sink]]：attention sink 现象专题
- [[Low-Rank KV Compression]]：低秩压缩方向
- [[Emergent Outlier Features]]：LLM outlier 特征
- [[Quantization Method Selection]]：量化方法选型
