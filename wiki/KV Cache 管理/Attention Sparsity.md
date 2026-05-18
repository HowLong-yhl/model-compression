---
title: Attention Sparsity
aliases: [注意力稀疏性, Sparse Attention, Attention Concentration]
tags: [model-compression, concept, attention, theory, kv-cache]
created: 2026-04-28
updated: 2026-04-28
sources: [H2O, SnapKV, KIVI, KVzip]
---

## 定义

Attention Sparsity 是指 Transformer 在推理时，softmax attention 权重高度集中于少数 token——即使模型是密集训练的，推理时的 attention 矩阵也有 **>95% 的权重接近零**。这一现象是 KV cache eviction 和 Value 量化的共同理论基础。

## 核心特征

### 经验性证据

**[[H2O]] 的幂律分布发现**：将所有 token 按累计 attention score 排序，分布呈幂律——极少数 "Heavy Hitter" token 持续获得不成比例的高 attention，其余 token 的 attention 贡献接近于零。

**[[SnapKV]] 的模式稳定性**：prompt 最后几个 token（observation window）的 attention 模式能稳定预测生成阶段的 attention 分配。这说明 sparsity 不是随机的——哪些 token "重要" 是高度可预测的。

**[[KVzip]] 的 reconstruction sparsity**：让 LLM 复述原始上下文时，reconstruction attention 比正常 prefill attention 更加稀疏，使得 context reconstruction 能更好地区分 KV 对的重要性。

### 数值特征

- Attention 矩阵中 >95% 的值接近零（[[H2O]]）
- 仅 ~5% 的 KV 对参与有效的 attention 计算
- Sparsity 随序列长度增加而增强（长上下文中更稀疏）
- 不同 attention head 的 sparsity 程度不同——某些 head 高度集中，其他 head 较均匀

## 两大应用方向

### 方向一：支撑 Token Eviction

Attention sparsity 是所有 [[Token Eviction]] 方法的理论基础——如果 >95% 的 token 对输出贡献接近零，那么移除它们的 KV cache 不会显著影响输出质量。

| 方法 | 如何利用 sparsity |
|------|------------------|
| [[H2O]] | 保留累计 attention 最高的 Heavy Hitter + 最近 token |
| [[SnapKV]] | 用 observation window 的 attention 投票选择重要 token |
| [[Scissorhands]] | 基于 pivotal token 的历史 attention 持久性 |
| [[KVzip]] | 用 reconstruction attention 的集中性评估 KV 重要性 |
| [[TriAttention]] | 用 pre-RoPE Q/K concentration 预测 attention 集中方向 |

### 方向二：保护 Value 量化

[[KIVI]] 利用 attention sparsity 论证了 Value per-token 量化的可行性：输出 $Y = AV$，当 attention 权重 $A$ 的大部分行接近零时，per-token Value 的量化误差只在低权重 token 上累积，不影响最终输出。

更形式化地：$\Delta Y = A \cdot E_v$，其中 $E_v$ 是 per-token 量化误差。当 $A_{:,i} \approx 0$（低权重 token），即使 $E_v[i]$ 较大也不影响 $\Delta Y$。

## 与 Emergent Outlier Features 的关系

[[Emergent Outlier Features]] 和 Attention Sparsity 是两个相关但不同的现象：
- **Outlier Features**：激活值中少数通道幅度极大（通道维度的不均匀性）
- **Attention Sparsity**：attention 权重中少数 token 权重极大（序列维度的集中性）

两者有因果联系：outlier channel 的存在会放大某些 token 对 attention score 的贡献（通过 Q·K 内积），从而加剧 attention 的稀疏性。[[StreamingLLM]] 的 [[Attention Sink]] 发现进一步揭示了这种联系——初始 token 的 sink 行为可能与其在 outlier channel 上的特殊值有关。

## 开放问题

1. **Sparsity 的来源**：attention sparsity 是训练产生的还是 softmax 本身的特性？softmax 的指数放大效应会将小差异放大为大 sparsity，但 H2O 的 heavy hitter 模式暗示这也与数据分布有关
2. **Sparsity 的可预测性上限**：TriAttention 的三角级数达到 r=0.72 的相关性——能否做到更高？
3. **Sparsity 与模型能力的关系**：更稀疏的 attention 是否意味着更弱的长程依赖？还是说 sparsity 是高效信息编码的标志？

## 引用关系

← 实证基础：[[H2O]]（幂律分布），[[SnapKV]]（模式稳定性），[[KVzip]]（reconstruction sparsity）
← 相关现象：[[Emergent Outlier Features]]（通道维度的集中性），[[Attention Sink]]（位置维度的特殊 token）
→ 应用方向：[[Token Eviction]]（eviction 的理论支撑），[[KV Cache Quantization]]（Value 量化的误差容忍）
→ 概念关联：[[KV Cache Management]]（sparsity 是 KV cache 管理的理论根基之一）
