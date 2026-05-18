---
title: "Q-VLM: Post-training Quantization for Large Vision-Language Models"
aliases: [Q-VLM, Shang 2024]
source_type: paper
source_url: "https://arxiv.org/abs/2410.04248"
source_date: 2024-10-06
ingested: 2026-04-28
venue: NeurIPS 2024
authors: [Changyuan Wang, Ziwei Wang, Xiuwei Xu, Yansong Tang, Jie Zhou, Jiwen Lu]
affiliation: Tsinghua University, S-Lab NTU
code: "https://github.com/ChangyuanWang17/QVLM"
tags: [model-compression, quantization, VLM, multimodal, PTQ, LLaVA, MoE-LLaVA, cross-layer-dependency, entropy]
---

## 核心要点

Q-VLM 是首个专门针对 **大型视觉-语言模型（LVLM）** 的后训练量化框架（NeurIPS 2024）。与直接将 LLM 量化方法迁移至多模态模型不同，Q-VLM 识别并解决了 LVLM 独有的核心挑战——**跨层依赖（cross-layer dependency）**：视觉编码器输出的量化误差会沿 pipeline 向下传播并与后续层的误差非线性叠加，导致逐层独立量化严重失效。

核心创新分为两部分：

1. **跨层依赖挖掘（Cross-Layer Dependency Mining）**：用激活熵（activation entropy）作为代理指标将网络层划分为强依赖块组（block），块内联合搜索 rounding function，在建模跨层依赖和保持可控搜索空间之间取得平衡。
2. **视觉编码器优化（Visual Encoder Optimization）**：解耦视觉编码器内部的跨层依赖，使其能被分解为更细粒度的独立子问题，进一步缩小搜索空间。

## 问题定义：为什么 LVLM 量化更难

### 跨层依赖

传统 LLM 量化假设各层可独立优化（layer-wise decomposition）。但在 LVLM 中，视觉编码器的输出经过 projector 后直接作为 LLM 的输入 token，视觉端的量化误差经多次变换后**以非线性方式累积**，使层间不再独立。

具体表现为：
- 视觉编码器中相邻层的量化误差高度相关
- LLM 部分对视觉 token 的量化误差呈放大效应
- 逐层独立量化（如直接套用 GPTQ/AWQ）在 W4A4 下性能退化严重

### 量化目标

对于 LVLM 的逐块量化，目标函数为最小化量化后输出与全精度输出的分布差异：

```
min E_D [D_KL(p(y|x_v, x_q) ‖ p_Q(y|x_v, x_q))]
```

其中 D_KL 为 KL 散度，作者将其分解为各 block 的 **Dependency-aware Expected Deviation (DED)**：

```
DED_i = E_D [‖O_i^Q - O_i‖₂] / E_D [‖O_i‖₂]
```

## 方法详解

### Part 1：跨层依赖挖掘

**目标**：将网络的 N 层划分为若干 block（每 block 最多 K 层），使 block 内部联合量化能捕捉跨层依赖，而 block 间可以独立处理。

**为什么选择熵作为代理（Proxy）？**

直接计算 DED 需要穷举所有 block 划分方案并逐一运行量化——组合爆炸，不可行。作者发现：

| 代理指标 | 与 DED 的相关系数 | 搜索开销 |
|---------|-----------------|---------|
| 量化误差 (Quantization Error) | 0.81 | 高（需运行量化） |
| **激活熵 (Activation Entropy)** | **0.97** | **低（仅需一次前向传播）** |

激活熵的定义：对第 i 层输出 $O_i$ 进行 softmax 归一化后计算 Shannon 信息熵。低熵意味着激活分布集中（尖峰/outlier），量化更困难。

**分块算法**：

1. 对每一层计算激活熵 $H_i$
2. 相邻层熵差 $\Delta H_i = |H_{i+1} - H_i|$ 衡量跨层依赖强度——$\Delta H$ 小说明两层的激活分布相似，依赖更强
3. 用动态规划（DP）将层序列划分为块组，使得：
   - 每块最多 K=3 层（控制搜索空间）
   - 块内 $\Delta H$ 之和最小化（强依赖的层被分在一起）
   - 块间切割点选在 $\Delta H$ 大的位置（弱依赖处切割）

**块内联合 Rounding**：

对每个 block，使用扩展的 rounding function 搜索：
- Rounding function 选择空间为 {Round-to-Nearest, ...}
- Block 内所有层的 rounding function 联合搜索，优化目标为 block 级 DED
- 通过 block-wise decomposition 将全模型 $O(2^N)$ 的搜索空间缩减到 $O(K \cdot 2^{N/K})$

### Part 2：视觉编码器优化

视觉编码器（通常为 ViT）内部的跨层依赖更加紧密（因为 ViT 层结构更均匀）。直接按上述方法分块后，搜索空间仍然较大。

**解耦策略**：

1. 分析视觉编码器中 Self-Attention 和 FFN 的依赖结构
2. 发现 FFN 层的跨层依赖相对较弱（因为 FFN 是 position-wise 的）
3. 将 FFN 从强依赖链中解耦出来，作为独立的单层 block 处理
4. 仅对 Self-Attention 层保留块内联合搜索

这种解耦使视觉编码器的搜索空间降低到原来的 $1/2^{K-1}$，同时几乎不损失精度。

## 实验结果

### 主实验：LLaVA-1.5 量化

#### W4A4（权重 4-bit，激活 4-bit）

| 方法 | ScienceQA | VizWiz | VQAv2 | HatefulMemes |
|------|-----------|--------|-------|-------------|
| **FP16 (LLaVA-7B)** | 80.23 | 54.00 | 78.50 | 60.20 |
| AWQ | 74.02 | 39.01 | 63.37 | 52.60 |
| QLoRA | 77.53 | 49.01 | 72.86 | 57.80 |
| ZeroQuant-V2 | 78.18 | 44.51 | 74.05 | 58.00 |
| **Q-VLM** | **79.79** | **50.15** | **76.23** | **58.60** |

Q-VLM 在所有任务上均优于 QLoRA（需微调）和 AWQ（纯 PTQ）。

#### W8A8

| 方法 | ScienceQA | VizWiz | VQAv2 | HatefulMemes |
|------|-----------|--------|-------|-------------|
| **FP16 (LLaVA-7B)** | 80.23 | 54.00 | 78.50 | 60.20 |
| SmoothQuant | 79.52 | 52.54 | 77.78 | 59.00 |
| **Q-VLM** | **80.13** | **53.72** | **78.32** | **59.80** |

W8A8 下 Q-VLM 几乎无损。

### LLaVA-13B W4A4

| 方法 | ScienceQA | VizWiz | VQAv2 | HatefulMemes |
|------|-----------|--------|-------|-------------|
| **FP16** | 80.96 | 63.33 | 80.01 | 63.40 |
| AWQ | 74.47 | 43.59 | 65.78 | 55.80 |
| QLoRA | 78.32 | 55.46 | 75.46 | 60.20 |
| **Q-VLM** | **80.78** | **58.43** | **78.14** | **61.80** |

13B 模型上 Q-VLM 在 ScienceQA 上仅比 FP16 差 0.18%。

### MoE-LLaVA W4A4

| 方法 | ScienceQA | VizWiz |
|------|-----------|--------|
| **FP16** | 78.02 | 42.97 |
| AWQ | 72.01 | 32.54 |
| **Q-VLM** | **76.75** | **40.48** |

在 Mixture-of-Experts 架构上同样有效。

### 效率收益（LLaVA-13B W4A4）

| 指标 | 数值 |
|------|------|
| 内存压缩 | **2.78×** |
| 推理加速 | **1.44×** |
| 量化时间 | ~2 GPU hours（单卡 A100） |

## Calibration

- **数据集**：LLaVA-Instruct-150K 中随机采样
- **样本数**：128 图文对
- **用途**：计算激活熵和 block-wise rounding function 搜索
- **搜索约束**：每 block 最多 K=3 层

## 消融实验

### 熵 vs 量化误差 vs 随机分块

| 分块策略 | ScienceQA (LLaVA-7B W4A4) |
|---------|--------------------------|
| 随机分块 | 77.82 |
| 量化误差引导 | 78.94 |
| **熵引导（Q-VLM）** | **79.79** |

熵引导分块比量化误差引导高 0.85，比随机分块高 1.97。

### Block Size K 的影响

| K | ScienceQA | 搜索时间 |
|---|-----------|---------|
| 1（逐层独立） | 78.18 | 1× |
| 2 | 79.14 | ~2× |
| **3** | **79.79** | ~4× |
| 4 | 79.82 | ~8× |

K=3 是精度-效率的最佳平衡点：K=4 仅比 K=3 高 0.03，但搜索时间翻倍。

### 视觉编码器解耦的效果

| 策略 | ScienceQA | 搜索空间 |
|------|-----------|---------|
| 不解耦 | 79.82 | 1× |
| **解耦 FFN** | **79.79** | **0.25×** |

解耦后搜索空间减少 75%，精度仅下降 0.03。

## 局限性

- **仅测试 LLaVA 家族**：未验证在 BLIP-2、InstructBLIP、Qwen-VL 等其他架构上的效果
- **依赖 calibration 数据**：需要图文对进行校准，不是 zero-shot 方法
- **Block size 是超参数**：K=3 在 LLaVA 上最优，其他架构可能不同
- **搜索开销**：虽然比穷举快，但仍需 ~2 GPU hours，不如 AWQ 的 grid search 快
- **未探索更低比特**：未测试 W2A4 或 W3A3 等极端低比特设定
- **视觉编码器解耦的通用性**：FFN 解耦假设是否对所有 ViT 变体成立尚未验证

## 与已有知识的关联

### 与 VLM Quantization Best Practices 的互补

[[VLM Quantization Best Practices]]（Maryland, 2026）从**分析角度**研究 VLM 组件敏感度，发现"组件敏感度与参数量不成正比"和"量化方法改变重要性分布"——但仅使用了现有 LLM 量化方法（[[GPTQ]]、[[AWQ]]）的直接迁移。

Q-VLM 则从**方法角度**提出了专门为 VLM 设计的量化框架，核心区别在于：

| 维度 | VLM Best Practices | Q-VLM |
|------|-------------------|-------|
| 定位 | 分析框架（哪个组件更重要） | 量化方法（如何更好地量化） |
| 量化范围 | Weight-only (W4A16) | Weight + Activation (W4A4) |
| 跨组件处理 | 独立量化各组件，事后分析交互 | 建模跨层依赖，联合优化 |
| 视觉编码器 | 发现 ViT 重要性被低估 | 专门优化 ViT 的量化策略 |

两篇论文的发现相互验证：Best Practices 发现"组件间存在非加性交互效应"——这正是 Q-VLM 的跨层依赖所建模的现象。

### 与 LLM 量化方法的关系

- **[[GPTQ]]**：Q-VLM 使用 block-wise rounding 替代 GPTQ 的 column-wise Hessian 补偿，因为后者假设层间独立
- **[[AWQ]]**：AWQ 的 per-channel scaling 可作为 Q-VLM 的预处理步骤，两者在原理上可组合
- **ZeroQuant-V2**：作为 baseline 之一，其 Low-rank 补偿在 VLM 上效果不如 Q-VLM 的跨层依赖方法

### 与扩散模型量化的联系

Q-VLM 面临的"跨层依赖"问题与 [[Diffusion Model Quantization]] 中的"时间步依赖"有结构相似性——都是**量化误差沿 pipeline 传播和累积**。[[ViDiT-Q]] 的 Static-Dynamic Channel Balance 和 Q-VLM 的 cross-layer dependency mining 都在试图解决"独立量化假设失效"的问题，只是一个沿时间维度、一个沿层维度。

### 在两步分解框架中的位置

从 [[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 的统一框架看，Q-VLM 的创新主要在 **Step 2（误差补偿）**——不是改进 rounding function 本身，而是改变了**搜索 rounding function 的粒度**，从 per-layer 扩展到 per-block。Step 1（预量化变换）方面 Q-VLM 未引入新的 rotation 或 scaling，这是潜在的改进方向。

## 引用关系

← 基线方法：[[AWQ]]、[[GPTQ]]（作为 VLM 量化 baseline）
← 相关概念：[[Post-Training Quantization]]（Q-VLM 属于 PTQ 范式）、[[Weight-Only Quantization]]（Q-VLM 超越 weight-only，实现 W4A4）
→ 互补分析：[[VLM Quantization Best Practices]]（VLM 组件敏感度分析，发现与 Q-VLM 的跨层依赖互相验证）
→ 结构类比：[[Diffusion Model Quantization]]（跨层依赖 vs 时间步依赖的结构相似性）
→ 应用场景：VLM 部署、多模态模型压缩、边缘推理
