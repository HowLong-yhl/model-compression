---
title: "OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models"
aliases: [OmniQuant]
source_type: paper
source_url: "https://arxiv.org/abs/2308.13137"
source_date: 2023-08-25
ingested: 2026-04-20
venue: ICLR 2024
authors: [Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, Ping Luo]
affiliation: Shanghai AI Lab / OpenGVLab / HKU
code: "https://github.com/OpenGVLab/OmniQuant"
tags: [model-compression, quantization, method, post-training, learnable, scaling, clipping]
---

## 核心要点

OmniQuant 是从 [[Unoptimized Scaling]]（SmoothQuant/AWQ）向 [[Optimized Scaling]]（OSTQuant/FlatQuant）演进的**关键桥梁方法**。它首次将等价缩放变换中的 scaling 因子和 weight clipping 参数作为**可学习参数**通过梯度下降优化，同时通过巧妙设计保持了 PTQ 的轻量和高效——仅需约 1-16 小时 GPU 时间即可完成量化。

## 方法摘要

### 核心创新：两组可学习参数

OmniQuant 引入了两个互补的可学习组件：

**1. Learnable Weight Clipping (LWC)**

对权重的量化范围上下界做可微参数化：

```
w_clip = clamp(w, -α, β)      α, β 为可学习参数
```

传统 min-max clipping 是全局最优但局部次优的；LWC 允许每个通道/group 学习独立的 clipping 范围，在 group-wise 量化中尤为有效。

**2. Learnable Equivalent Transformation (LET)**

对 [[Equivalent Scaling Transformation]] 中的缩放因子 s 做可学习参数化：

```
Y = (X · diag(s)⁻¹) · (diag(s) · W)    s 为可学习参数（非公式计算）
```

与 SmoothQuant/AWQ 的区别：
- SmoothQuant 用 `s = max(|X_j|)^α / max(|W_j|)^{1-α}` 封闭公式
- AWQ 用 `s = s_X^α` + grid search
- **OmniQuant 让 s 自由初始化并通过梯度下降端到端优化**

此外还引入了 shifting 参数（learnable zero-point offset），对应综述论文分类中的 "Shifting + Scaling"。

### 优化策略

| 设计选择 | OmniQuant 方案 | 设计意图 |
|----------|---------------|---------|
| 优化目标 | Block-wise knowledge distillation（对齐 FP16 层输出） | 避免全模型 loss 的高内存开销 |
| 可学习参数量 | 仅 s, α, β（极少量）| 保持 PTQ 轻量特性 |
| 权重梯度 | 不更新原始权重 | 仍是 PTQ 而非 QAT |
| STE | 用于量化操作的梯度近似 | 使 clipping/scaling 可微 |
| 训练时间 | LLaMA-7B 约 1 小时（1×A100） | 与 GPTQ 量化时间可比 |

### 在两步分解框架中的位置

根据 [[A Comprehensive Evaluation on Quantization Techniques for LLMs]]：

```
OmniQuant = Shifting + Scaling (optimized) + GPTQ
```

它是第一个将 Step 1（预量化变换）从公式/搜索升级为梯度优化的方法，但其 Step 2 仍使用标准 GPTQ。

## 实验结果

### WikiText-2 Perplexity (W4A16, group-128)

| 模型 | FP16 | RTN | GPTQ | AWQ | **OmniQuant** |
|------|------|-----|------|-----|---------------|
| LLaMA-7B | 5.68 | 5.96 | 6.22 | 5.78 | **5.73** |
| LLaMA-13B | 5.09 | 5.20 | 5.17 | 5.19 | **5.12** |
| LLaMA-30B | 4.10 | 4.24 | 4.20 | 4.19 | **4.14** |

### W2A16 极端低比特 (group-64)

| 模型 | RTN | GPTQ | **OmniQuant** |
|------|-----|------|---------------|
| LLaMA-7B | 5e4 | 12.23 | **9.72** |
| LLaMA-13B | 3e4 | 8.23 | **7.12** |

OmniQuant 在 2-bit 极端场景下大幅领先 GPTQ——LWC 的自适应 clipping 在 bit 极低时收益尤为显著。

### W4A4 量化——Zero-shot 任务精度

Weight-Activation 联合量化是 OmniQuant 相较前作的重大突破。W4A4 使用 per-channel weight + per-token activation 量化，Softmax 保持 FP16：

| 模型 | 方法 | PIQA | ARC-e | ARC-c | BoolQ | HellaSwag | Winogrande | Avg. |
|------|------|------|-------|-------|-------|-----------|------------|------|
| LLaMA-7B | SmoothQuant | 49.80 | 30.40 | 25.80 | 49.10 | 27.40 | 48.00 | 38.41 |
| | OS+ | 62.73 | 39.98 | 30.29 | 60.21 | 44.39 | 52.96 | 48.43 |
| | LLM-QAT | 55.90 | 35.50 | 26.40 | 62.40 | 47.80 | 50.60 | 46.43 |
| | **OmniQuant** | **66.15** | **45.20** | **31.14** | **63.51** | **56.44** | **53.43** | **52.65** |
| LLaMA-13B | SmoothQuant | 61.04 | 39.18 | 30.80 | 61.80 | 52.29 | 51.06 | 49.36 |
| | OS+ | 63.00 | 40.32 | 30.38 | 60.34 | 53.61 | 51.54 | 49.86 |
| | **OmniQuant** | **69.69** | **47.39** | **33.10** | **62.84** | **58.96** | **55.80** | **54.37** |

OmniQuant W4A4 在 LLaMA-7B 上达到 52.65% 平均精度——以 PTQ 效率达到了接近 QAT（LLM-QAT: 46.43%）的性能。SmoothQuant 在 W4A4 下几乎崩溃（38.41%），说明手工迁移强度在极低比特下严重不足。

在 W4A4 下 OmniQuant 仍不如后续的旋转方法（QuaRot、SpinQuant 等可进一步降至 ~7-8 PPL），说明仅 Scaling+Shifting 对 outlier 的消除不够彻底。

### W6A6 量化

| 模型 | 方法 | Avg. 精度 |
|------|------|----------|
| LLaMA-7B | SmoothQuant | 62.81 |
| | OS+ | 61.13 |
| | **OmniQuant** | **63.17** |
| LLaMA-65B | SmoothQuant | 69.80 |
| | OS+ | 68.76 |
| | **OmniQuant** | **70.28** |

W6A6 下 OmniQuant 在 65B 模型上甚至略优于 FP16 SmoothQuant（70.28 vs 69.80），说明梯度优化的等价变换在中等比特下效果显著。

## 与已有等价变换方法的对比

| 方法 | 变换操作 | 应用位置 | 参数确定方式 | 适用场景 |
|------|---------|---------|-------------|---------|
| [[SmoothQuant]] | Scaling | Linear 层 | 公式预定义 | W-A 量化 |
| [[AWQ]] | Scaling | Linear 层 | Grid search | Weight-only |
| OS+ | Scaling + Shifting | Linear 层 | Grid search (scaling) + 公式 (shifting) | W-A 量化 |
| **OmniQuant** | **Scaling + Shifting** | **Linear 层 + Attention** | **梯度优化** | **Weight-only + W-A** |

OmniQuant 在三个维度上扩展了前作：操作上加入了 shifting，位置上扩展到了 attention 内部的 Q/K 矩阵乘法（虽然收益较小，因 outlier 主要在 normalization 输出），参数确定上用端到端梯度取代了启发式搜索。

## 消融实验

### 组件效果

| 配置 | LLaMA-13B W4A4 | OPT-13B W4A4 |
|------|----------------|-------------|
| LWC + LET（完整） | **10.87** | **11.65** |
| 仅 LET（去掉 LWC） | 20.75 | 15.23 |
| 仅 LWC（去掉 LET） | 5.4e3 | 7.8e3 |
| 无（去掉两者） | 1.8e3 | 7.8e5 |

LET 对 W4A4 至关重要——去掉后 PPL 爆炸到数千。这是因为 W4A4 下激活的 outlier 是主要难点，LET 的等价变换直接解决了这个问题。LWC 在 LET 之上进一步降低 40-50% 的 PPL。

值得注意的是，对 LLaMA 的 **weight-only 量化**，LET 几乎无效（因 LLaMA 的权重 outlier 很少），因此在 weight-only 设定下 OmniQuant 只启用 LWC 以加速训练。

### LET 各位置的影响

| 去掉的 LET 位置 | LLaMA-7B W4A4 Avg. PPL |
|----------------|----------------------|
| 完整 | **12.87** |
| 去掉 ln1→(q,k,v proj) | 19.87 |
| 去掉 v_proj→out_proj | 13.03 |
| 去掉 Q,K（attention 内） | 13.34 |
| 去掉 ln2→fc1 | 14.47 |

ln1→(q,k,v proj) 是最关键的 LET 位置（去掉后 PPL 从 12.87 涨到 19.87），因为大部分 outlier 出现在 normalization 输出处。

### Softmax 量化的影响

| Softmax 精度 | LLaMA-7B W4A4 Avg. PPL | Avg. Acc. |
|-------------|----------------------|-----------|
| 16-bit | 12.87 | 52.65% |
| 8-bit | 12.91 | 51.93% |
| 6-bit | 13.20 | 51.70% |
| 4-bit | 18.80 | 48.52% |

Softmax 的长尾分布使得 4-bit 量化带来显著退化，但 8-bit/6-bit 可接受——block-wise 校准能补偿部分损失。

### Calibration 数据鲁棒性

| Calibration 来源 | W3A16 WikiText2 PPL | W4A4 WikiText2 PPL |
|-----------------|--------------------|--------------------|
| WikiText2 | 6.47 | 11.23 |
| C4 | 6.67 | 12.17 |
| Pile | 6.69 | 12.04 |
| **方差** | **0.009** | **0.17** |

跨数据集方差极小（0.009-0.17），表明 OmniQuant 的梯度优化对 calibration 分布高度鲁棒。仅 16 个样本即可收敛。

## 部署与训练效率

### MLC-LLM 部署（A100-80G）

| 模型 | FP16 Memory | W4A16g128 Memory | FP16 Token/s | W4A16g128 Token/s |
|------|-----------|-----------------|-------------|-------------------|
| LLaMA-7B | 14.4G | 5.7G | 69.2 | **134.2** |
| LLaMA-13B | 27.1G | 10.0G | 52.5 | **91.3** |
| LLaMA-65B | OOM | 41.0G | — | **24.3** |

LWC 和 LET 在量化后**完全融入权重**，推理时零额外开销，因此 OmniQuant 可直接利用现有的 W4A16 推理引擎（如 MLC-LLM）。LLaMA-65B 在 FP16 下 OOM，量化后可在单卡运行。

### 训练时间（单卡 A100-80G，128 样本，20 epochs）

| 模型 | Weight-only | Weight-Activation |
|------|-----------|-------------------|
| LLaMA-7B | **1.1h** | 1.6h |
| LLaMA-13B | 2.2h | 3.3h |
| LLaMA-30B | 4.5h | 7.3h |
| LLaMA-65B | 8.9h | 14.4h |

### Instruction-tuned 模型（LLaMA-2-Chat，Vicuna-Bench）

在 W3A16g128 下，OmniQuant 在 LLaMA-2-7b-chat 上 win rate 为 50%（vs AWQ）和 80.3%（vs RTN）；在 LLaMA-2-13b-chat 上持续优于 RTN 和 AWQ。证明了在 instruction-tuned 模型上的泛化性。

### Scaling Law 观察

OmniQuant 使得 **3-bit 量化在 perplexity-total bits 权衡上达到了与传统 4-bit 量化相当的性能**。这意味着在给定内存预算下，3-bit OmniQuant 可能优于 4-bit RTN/GPTQ。

```
公式缩放 (SmoothQuant, AWQ)
        ↓ 将 s 从公式变为可学习参数
学习缩放 (OmniQuant)
        ↓ 加入旋转，联合优化正交+缩放
联合优化 (OSTQuant, FlatQuant)
```

OmniQuant 证明了两个关键命题：
1. **可学习 > 公式**：即使用同样的缩放变换形式，梯度优化比 grid search 找到更好的 s
2. **Scaling alone 有上限**：W4A4 下 OmniQuant 的 PPL（~13）远不如后续旋转方法（~7-8），说明仅缩放不够——需要引入旋转来彻底消除 outlier

## 局限性

- **仅做 Scaling + Shifting**：不做 Rotation，对 W4A4 的 outlier 消除不够彻底
- 需要梯度计算：比 SmoothQuant/AWQ 的纯前向方法更慢（但比 QAT 快很多）
- Block-wise distillation 可能引入误差累积（各 block 独立优化，无全局协调）
- 在第三代方法（SpinQuant、FlatQuant）面前精度已不是 SOTA

## 与其他方法的关系

- **vs [[SmoothQuant]] / [[AWQ]]**：共享 [[Equivalent Scaling Transformation]] 的形式，但从公式/搜索升级为梯度优化。证明了 learned s > formula s
- **vs [[GPTQ]]**：OmniQuant 的 Step 2 仍使用 GPTQ，两者是正交组合关系
- **vs [[SpinQuant]] / [[OSTQuant]] / [[FlatQuant]]**：后续方法在 OmniQuant 的基础上引入了 Rotation，实现了更彻底的 outlier 消除和更优的 W4A4 性能
- **vs [[Unoptimized Scaling]]**：OmniQuant 是 unoptimized → optimized scaling 的转折点

## 引用关系

← 承接：[[SmoothQuant]]（等价变换的思想），[[AWQ]]（激活感知缩放的形式）
← 使用：[[GPTQ]]（作为 Step 2 误差补偿）
→ 启发了：[[FlatQuant]]（进一步引入 Kronecker 可学习变换），[[OSTQuant]]（联合优化理论化）
→ 被集成：[[SliM-LLM]]（SliM-LLM+ 将 SBA 混合精度分配集成到 OmniQuant 的 LWC 框架中，进一步降低极低比特下的 PPL）
→ 分类：[[Optimized Scaling]]（可学习缩放的首个代表）
