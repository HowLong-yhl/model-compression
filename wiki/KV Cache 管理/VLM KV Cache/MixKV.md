---
title: "MixKV: Mixing Importance with Diversity for KV Cache Compression in Large Vision-Language Models"
aliases: [MixKV, Mixing Importance with Diversity]
source_type: paper
source_url: "https://arxiv.org/abs/2501.07002"
source_date: 2025-01-28
ingested: 2026-05-14
venue: ICLR 2026
authors: [Xuyang Liu, Xiyan Gui, Yuchao Zhang, Linfeng Zhang]
affiliation: EPIC Lab, Shanghai Jiao Tong University / Sichuan University / HUST
code: "https://github.com/xuyang-liu16/MixKV"
tags: [model-compression, kv-cache, eviction, VLM, diversity, importance, head-wise, adaptive, plug-and-play, semantic-redundancy]
---

## 核心要点

MixKV 提出了一个关键洞察：**仅基于 importance 的 KV cache eviction 在 VLM 中会导致语义覆盖丢失**。因为 VLM 中高 importance 的 KV pairs 往往彼此高度相似（语义冗余），保留它们虽然维持了"重要性"，但丢失了信息分布的多样性。MixKV 通过 **混合 importance 与 diversity** 来优化 KV cache 压缩——自适应地在每个 attention head 上平衡这两个目标。

核心公式：$s_i^{comp} = (1-\bar{r}_h^l) \cdot s_{imp,i} + \bar{r}_h^l \cdot s_{div,scaled,i}$

其中 $\bar{r}_h^l$ 是该 head 的语义冗余度——冗余度高的 head 强调 diversity，冗余度低的 head 强调 importance。

![[images/MixKV_fig1_redundancy.png]]

*Figure 1: VLM (Qwen2-VL) 的 KV cache 比纯 LLM (Qwen2) 具有显著更高的语义冗余——Key 的平均余弦相似度从 0.2-0.4 升至 0.6-0.8（2-3× 增加）。*

## 两个关键发现

### 发现一：VLM 比 LLM 语义冗余更高

VLM 处理视觉信息时，由于图像中大量重复视觉元素（相似纹理、重复模式），KV pairs 间的语义相似度远高于纯文本场景。Figure 1(b) 量化显示：Qwen2 的 Key 平均相似度分布峰值在 0.2-0.4，而 Qwen2-VL 峰值在 0.6-0.8。

### 发现二：Head 间冗余度差异极大

![[images/MixKV_fig2_head_similarity.png]]

*Figure 2: 不同 attention head 的平均余弦相似度差异极大——某些 head 超过 0.9（高度冗余），另一些低于 0.3。这种模式跨任务一致。*

不同 head 关注多模态输入的不同方面：有的捕获全局特征（低冗余），有的关注局部细节（高冗余）。这意味着统一的 eviction 策略必然次优——高冗余 head 需要 diversity-focused selection，低冗余 head 适合 importance-focused selection。

### Importance-only 的缺陷

![[images/MixKV_fig3_tsne.png]]

*Figure 3: t-SNE 可视化——SnapKV (蓝色星) 仅覆盖 Full KV (灰色圆) 分布的一小部分，而 MixKV (红色圆) 覆盖范围更广。*

基于 importance 的方法倾向保留语义相似的"高分"KV pairs，导致压缩后的 cache 只覆盖原始信息分布的子集——丢失了全局语义覆盖。

## 方法

### Importance Score 设计

MixKV 整合了 **Extrinsic Importance**（外在重要性）和 **Intrinsic Importance**（内在重要性）：

- **Extrinsic**：observation window（默认最后 32 tokens）的平均 attention score——反映与 instruction 的相关性
- **Intrinsic**：Value Norm (ℓ₂ norm of value vectors)——反映 value 对 attention 输出的贡献大小

整合方式：$s_{imp} = s_{imp}^{ex} + s_{imp}^{in(VNorm)}$（VNorm 归一化后缩放至与 attention score 同量级）

消融验证：VNorm 优于 KNorm（Key 的 ℓ₂ norm），因为 VNorm 更好地捕获了 value 对多模态 attention 的贡献。

### Diversity Score 设计

基于 Key 向量与全局均值的负余弦相似度：

1. 归一化 Key：$\hat{K}_{h,i}^l = K_{h,i}^l / \|K_{h,i}^l\|$
2. 计算全局均值：$\bar{\hat{K}}_h^l = \frac{1}{T} \sum_{i=1}^T \hat{K}_{h,i}^l$
3. Diversity score：$s_{div,i} = -\hat{K}_{h,i}^l \cdot \bar{\hat{K}}_h^l$

直觉：与全局 pattern 越不同的 KV pair，diversity 越高——保留这些 pair 有助于维持信息分布的广度。计算复杂度为 O(T)，线性时间。

### Head-wise 冗余度量化

利用归一化 Key 矩阵，通过数学推导高效计算 off-diagonal 平均相似度：

$$\bar{r}_h^l = \frac{T^2 \|\bar{\hat{K}}_h^l\|_2^2 - T}{T(T-1)}$$

该值范围 [0, 1]：
- $\bar{r}_h^l \to 1$：高冗余 head → 增加 diversity 权重
- $\bar{r}_h^l \to 0$：低冗余 head → 增加 importance 权重

### Head-wise Adaptive Mixing

最终综合评分：

$$s_i^{comp} = (1 - \bar{r}_h^l) \cdot s_{imp,i} + \bar{r}_h^l \cdot s_{div,scaled,i}$$

其中 $s_{div,scaled,i}$ 经过 [0,1] 归一化后缩放至与 importance score 同量级。按 $s_i^{comp}$ 排序取 Top-B 即得压缩后的 KV cache。

**关键性质**：MixKV 是 plug-and-play 框架，可叠加到任何 importance-based 方法之上（SnapKV、PyramidKV、AdaKV、SparseMM），仅修改评分函数，不改变压缩算子。

## 实验结果

### 多模态理解（Table 1）

![[images/MixKV_table1_main_results.png]]

在 LLaVA-NeXT-Mistral-7B、InternVL3-8B、Qwen2-VL-7B 三种架构上，5 个 benchmark（DocVQA、OCRBench、TextVQA、ChartQA、TextCaps），Budget = 64/128/256：

**Budget=64 时 MixKV 对 baseline 的提升（LLaVA-NeXT-Mistral-7B）**：
| Base Method | DocVQA | OCRBench | TextVQA | ChartQA | TextCaps |
|-------------|--------|----------|---------|---------|----------|
| SnapKV → +MixKV | +1.5 | +4.2 | +3.0 | +0.9 | +0.070 |
| PyramidKV → +MixKV | +1.7 | +2.9 | +3.0 | +0.5 | +0.059 |
| AdaKV → +MixKV | +2.1 | +3.8 | +2.7 | +0.6 | +0.069 |
| SparseMM → +MixKV | +1.6 | +3.3 | +1.6 | +1.7 | +0.051 |

**极端压缩 (Budget=64) 下平均提升 5.1%**，高压缩比下优势更显著。

### GUI Grounding (Table 2)

Qwen2.5-VL-7B + ScreenSpot-v2：
- SnapKV+MixKV (budget=128)：75.3% → **83.3%** (+7.9%)
- AdaKV+MixKV (budget=64)：53.7% → **62.7%** (+9.0%)

### LLM 文本任务 (Table 3)

LongBench 上 Mistral-7B / Llama-3.1-8B：
- MixKV 同样带来一致但较小的提升（平均 +0.3 to +0.8）
- 在 Summarization（信息聚合）类任务提升更大，在 Localization 类任务偶有下降——因为 diversity 可能稀释局部检索的注意力

### 推理效率 (Figure 4)

![[images/MixKV_fig4_efficiency.png]]

*Figure 4: MixKV 的额外计算开销 <1% latency 增加，因冗余度计算 $\bar{r}_h^l$ 仅需 O(T) 操作。*

在 Qwen2-VL-7B、32K context、budget=64 下：
- MixKV 与底层 baseline 的 latency 和 peak memory 几乎无差异
- 相比 Full KV cache 有显著的 latency 和 memory 降低

## 消融实验关键结论

1. **VNorm > KNorm** 作为 intrinsic importance（KNorm 关注 Key 幅度，与 value-driven attention 动力学不对齐）
2. **纯 diversity 不可行**——严重破坏原始语义信息，性能大幅下降
3. **s_imp + s_div（无 head-wise 自适应）** 优于 s_imp alone，但不如 Whead 自适应
4. **Online per-sample r̄ > offline 固定 r̄**——per-sample 计算允许适配不同输入的冗余模式
5. **PyramidKV 在 VLM 中依然弱于 SnapKV**（与 VL-Cache、AirCache 发现一致）

## 与相关方法的关系

- **[[VL-Cache]]**：VL-Cache 发现 modality boundary + post-vision attention scoring；MixKV 发现 head-wise 语义冗余差异 + importance/diversity 混合。两者视角互补——VL-Cache 关注"哪些 token 重要"（scoring policy），MixKV 关注"保留的 tokens 是否覆盖了足够多样的信息"（selection diversity）
- **[[AirCache]]**：AirCache 关注 observation window 的选择质量（elite tokens 更一致），MixKV 关注被选中 tokens 的信息分布（diversity）。两者可能组合：AirCache 的 elite window 给出更好的 importance score，MixKV 的 diversity mixing 确保选择结果不过于冗余
- **[[SnapKV]]**：MixKV 的直接改进对象之一。SnapKV 用 observation window voting 的 importance-only 评分，MixKV 在此基础上混入 diversity
- **[[PyramidKV]]**：MixKV 同样可叠加于 PyramidKV，验证了 layer-level budget + head-level diversity 的组合有效性
- **[[DuoAttention]]**：DuoAttention 做 head-level binary classification，MixKV 做 head-level continuous redundancy weighting——粒度和思路不同但都发现了 head 异质性
- **[[Token Eviction]]**：MixKV 是 eviction 范式的 meta-improvement——不提出新的 eviction 方法，而是改进任意 importance-based 方法的评分函数
- **SparseMM (Wang et al., 2025)**：head importance-based 非对称 budget 分配的 VLM 方法，MixKV 可在其之上继续提升

## 对 VLM KV Cache 研究的启示

1. **语义多样性是被忽视的维度**：importance-only eviction 在 VLM 中尤其脆弱，因为视觉冗余让 top-importance tokens 彼此高度相似
2. **Head-wise 自适应是必要的**：不同 head 的冗余模式差异 2-3×，全局统一策略必然次优
3. **Per-sample 自适应优于 offline 统计**：不同输入（高分辨率 vs 低分辨率，文字密集 vs 场景图）的冗余模式不同
4. **Plug-and-play 框架思路**：MixKV 证明了"改进评分函数而不改变压缩流程"是可行且有效的方法论

## 局限

- Budget=64 极端压缩下仍有不小性能损失（如 DocVQA 从 93.9% 降至 67.9%）
- 在 LLM 纯文本任务上提升较小（因为 LLM 本身冗余度低于 VLM）
- 未探索与 KV cache quantization 的组合
- Diversity score 仅基于 Key（未考虑 Value 的 diversity）
- 未在视频 VLM 上评估长视频场景

## 引用关系

← 直接改进：[[SnapKV]]、[[PyramidKV]]、AdaKV、SparseMM
← 理论基础：KV cache 语义冗余的系统性分析
← 同方向：[[VL-Cache]]（modality-aware scoring）、[[AirCache]]（elite observation window）
→ 父概念：[[VLM KV Cache]]、[[Token Eviction]]
→ 与量化正交：可与 [[KV Cache Quantization]] 组合（尚未探索）
→ 代码开源：https://github.com/xuyang-liu16/MixKV
