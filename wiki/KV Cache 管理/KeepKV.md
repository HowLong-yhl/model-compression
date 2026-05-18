---
title: "KeepKV: Achieving Periodic Lossless KV Cache Compression for Efficient LLM Inference"
aliases: [KeepKV, Electoral Votes, ZIP-Merging]
source_type: paper
source_url: "https://arxiv.org/abs/2504.09936"
source_date: 2025-11-27
ingested: 2026-05-11
venue: AAAI 2026
authors: [Yuxuan Tian, Zihan Wang, Yebo Peng, Aomufei Yuan, Zhiming Wang, Bairen Yi, Xin Liu, Yong Cui, Tong Yang]
affiliation: Peking University / ByteDance / Tsinghua University
code: "https://github.com/pku-yuxuan/KeepKV"
tags: [model-compression, kv-cache, merging, eviction, token-compression, inference, long-context, lossless]
---

## 核心要点

KeepKV 是首个实现**单步无损 KV cache 压缩**并为多步压缩提供**理论误差界**的 KV cache merging 方法。核心洞察：现有 merging 方法（如 D2O、KVMerger）使用凸组合（weighted average）合并 KV 对，不可避免地导致 **Attention Sag**——合并后 KV 对的注意力分数低于原始多个 KV 对的分数之和，造成输出扰动。

KeepKV 通过两个创新设计消除此问题：
1. **Electoral Votes 机制**：记录每个 KV 对被合并的次数，用投票数缩放注意力分数，使合并后的 KV 对能正确反映它所代表的多个原始 KV 对的总影响
2. **Zero Inference-Perturbation Merging (ZIP-Merging)**：自动调整合并权重以补偿注意力损失，理论上保证当前步输出零扰动

结果：仅保留 **10% KV cache** 仍能维持生成质量，推理吞吐提升 **2×** 以上。

## 问题分析：Eviction 与 Merging 的局限

### Eviction 的不可逆信息损失

Eviction 方法丢弃被认为不重要的 (k_e, v_e)，导致输出偏移：

```
o'_t = 1/(1 - A^t_e) · (o_t - A^t_e · v_e)
```

偏移量由 A^t_e（被驱逐 token 的注意力分数）决定。虽然方法们（[[H2O]]、[[SnapKV]]）优先驱逐低分 token，但**无法消除**扰动。且被驱逐 KV 可能在后续生成中变得重要——不可逆丢失导致 hallucination。

### Merging 的 Attention Sag

现有 merging 方法将 (k_e, v_e) 合并入 (k_c, v_c)：

```
k_r = w_e·k_e + w_c·k_c,  v_r = w_e·v_e + w_c·v_c  (w_e + w_c = 1)
```

**Theorem 2**：凸组合 merging 必然导致合并后 KV 对的注意力分数**低于**原始两个分数之和（A'_r < A_e + A_c），从而 ‖o'_t - o_t‖ > 0。

直观解释：凸组合将多个向量融合为一个，后续注意力计算中该向量与其他单一向量同等对待——丢失了"它代表多个 token"的历史信息。

## 方法：KeepKV

### Electoral Votes 机制

为每个 KV 对维护一个投票计数 p_i（初始化为 1）。当 (k_e, v_e) 被合并入 (k_c, v_c) 时，结果 KV 对的投票数为 p_r = p_e + p_c。注意力计算时，每个 KV 对的分数乘以其投票数：

```
o_t = Σ(p_i · s^t_i · v_i) / Σ(p_i · s^t_i)
```

类比美国选举人团制度：投票数高的 KV 对（代表更多原始 token）在注意力中获得更大影响力。

### ZIP-Merging（Zero Inference-Perturbation Merging）

在 Electoral Votes 基础上，ZIP-Merging 设计了特殊的合并权重确保零输出扰动：

```
k_r = (w_e·k_e + w_c·k_c) · ln(p_e·s_e + p_c·s_c) / (w_e·ln(s_e) + w_c·ln(s_c))
v_r = (p_e·s_e·v_e + p_c·s_c·v_c) / (p_e·s_e + p_c·s_c)
```

其中权重为：w_e = p_e·s_e / (p_e·s_e + p_c·s_c), w_c = p_c·s_c / (p_e·s_e + p_c·s_c)

**Theorem 3**：此合并方法保证 ‖o'_t - o_t‖ = 0（当前步零扰动）。

关键点：Value 按注意力权重加权平均（保持输出不变），Key 的合并则通过对数缩放确保 softmax 后的注意力分数等于原始两个分数之和。

### 多步生成扩展：EMA 注意力分数预测

ZIP-Merging 需要知道当前步的注意力分数 s^t_i 来计算权重。对于未来步，使用 **Exponential Moving Average (EMA)** 预测注意力分数：

```
S_t = α·S_{t-1} + (1-α)·s_t  (t > L)
ŝ_t = S_t / (1 - α^t)  (bias correction)
```

**Theorem 5（多步误差界）**：若 EMA 预测误差满足 |1 - ŝ^t_i/s^{t'}_i| ≤ ε (ε < 1)，则输出扰动 Θ_{t'} < 2ε(1+ε)γ / (1-ε)²，其中 γ 为 value 向量间最大距离。

**Lemma 6**：当 ε → 0（预测完美）或被合并的 KV 对完全相同时，扰动趋向零。这为基于相似度的 merge candidate 选择提供了理论基础。

### Merge Candidate 选择

基于 Lemma 6 的理论指导：合并越相似的 KV 对，多步扰动越小。因此选择与待驱逐 KV 具有最高 Key cosine 相似度的保留 KV 作为合并对象，且设置阈值 T=0.8——低于阈值则不合并（退化为 eviction）。

### 与现有方法的组合

KeepKV 对 token selection 和 cache allocation 策略**无约束**——可以直接与任何 eviction 方法组合：
- 用 H2O/SnapKV/PyramidInfer 的策略决定哪些 token 要驱逐
- 用 KeepKV 的 Electoral Votes + ZIP-Merging 将驱逐集合并入保留集
- 实验证明组合后所有方法在所有压缩率下均获得提升

## 实验结果

### 极限压缩能力（HELM/lm-eval）

在 10% KV cache budget（90% 压缩）下，KeepKV 仍维持接近 full cache 的质量，远超所有 baseline。

### LongBench（20% 压缩率）

| 模型 | 方法 | Single-Doc QA | Multi-Doc QA | Summarization | Synthetic | Code |
|------|------|---|---|---|---|---|
| Llama-3-8B | Full | 14.34/13.68/21.70 | 9.42/10.75/6.99 | 45.13 | 3.74/6.72 | 70.54/66.04 |
| | H2O | 13.73/10.02/17.20 | 9.31/10.62/6.42 | 45.02 | 3.29/7.56 | 68.95/63.84 |
| | D2O | 13.50/8.86/17.21 | 9.16/10.52/6.35 | 44.64 | 3.44/5.80 | 68.49/64.84 |
| | **KeepKV** | **12.76/10.63/18.57** | **9.37/10.72/6.53** | **45.20** | **3.54/7.16** | **69.05/65.68** |

KeepKV 在 Summarization 和 Multi-Doc QA 上尤其突出（这些任务需要完整上下文，注意力不稀疏）。

### 推理吞吐量

| 方法 | Batch Size | 吞吐 (tok/s) | 相对 Full |
|------|-----------|------------|---------|
| Full cache | 2 | 116.54 | 1× |
| H2O (eviction) | 8 | 317.33 | 2.72× |
| D2O (merging) | 8 | 214.80 | 1.84× |
| **KeepKV** | 8 | **255.99** | **2.20×** |

Merging 方法比纯 eviction 慢（需额外计算），但 KeepKV 比 SOTA merging (D2O) 快 19%，因为不需要动态 cache allocation 的额外开销。

## 理论贡献

KeepKV 的核心价值在于为 KV cache merging 建立了**理论框架**：

1. **Eviction 扰动定理**：量化了 eviction 导致的不可消除输出偏移
2. **Attention Sag 定理**：证明凸组合 merging 必然低估合并 KV 的注意力贡献
3. **ZIP-Merging 无损定理**：证明当前步可实现零扰动
4. **多步误差界**：将多步扰动与 EMA 预测误差和 merge 对象相似度建立定量关系
5. **相似度选择的理论基础**：为 D2O/KVMerger 等基于相似度的 candidate selection 提供了首次理论解释

## 与其他 KV Cache 管理方法的关系

| 方法类别 | 代表 | 信息保留 | 硬件友好 | KeepKV 关系 |
|---------|------|---------|---------|-----------|
| Eviction | [[H2O]], [[SnapKV]] | 低（不可逆丢弃） | 高 | KeepKV 可替代/增强 eviction |
| Quantization | [[KIVI]], [[KVQuant]], [[KVTuner]] | 高（降精度保留） | 高 | 正交互补（可组合） |
| Intra-layer Mixed | KIVI prefix/recent | 中 | 中 | 正交互补 |
| Merging (凸组合) | D2O, KVMerger, CaM | 中（Attention Sag） | 高 | KeepKV 取代（理论更优） |
| **KeepKV** | — | **高（理论无损）** | **高** | — |

### 与量化的正交性

KeepKV 减少 KV cache 的 **token 数量**，量化减少每个 token 的 **比特数**。二者可叠加：如 KeepKV 压缩到 20% token + KIVI 4-bit 量化 = 理论 ~40× 压缩。

## Periodic Lossless Compression 模式

KeepKV 还支持**周期性无损压缩**：将完整 KV cache 存储在外部（CPU/SSD），周期性地将压缩表示加载到 GPU 显存中推理。由于 ZIP-Merging 保证加载时刻的零扰动，这使得 KeepKV 成为 KV cache offloading 场景的理想压缩方案。

## 关联页面

- [[KV Cache Quantization]]——KeepKV 与量化正交互补
- [[Token Eviction]]——KeepKV 改进 eviction 为 merging，消除不可逆信息损失
- [[H2O]]——Heavy Hitter eviction，可与 KeepKV 组合使用
- [[SnapKV]]——观察窗口投票 eviction，可与 KeepKV 组合使用
- [[Attention Sparsity]]——KeepKV 的理论分析基于注意力分布的 locality 和 sparsity 特性
- [[KV Cache Offloading]]——KeepKV 的周期性无损压缩可用于 offloading 场景
- [[KVTuner]]——layer-wise 精度分配，与 KeepKV 的 token-level merging 正交
