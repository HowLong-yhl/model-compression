---
title: "Towards Understanding Best Practices for Quantization of Vision-Language Models"
aliases: [VLM Quantization Best Practices, MMQ]
source_type: paper
source_url: "https://arxiv.org/abs/2601.15287"
source_date: 2026-01-21
ingested: 2026-04-28
venue: Preprint (under review)
authors: [Gautom Das, Vincent La, Ethan Lau, Abhinav Shrivastava, Matthew Gwilliam]
affiliation: University of Maryland, College Park
code: "https://github.com/gautomdas/mmq"
tags: [model-compression, quantization, VLM, multimodal, BLIP-2, LLaVA, GPTQ, AWQ, component-sensitivity, PTQ]
---

## 核心要点

本文是首个系统研究 **多模态大语言模型（MLLM）量化** 的工作，回答一个实际部署中的核心问题：当我们量化一个 VLM pipeline（ViT + Connector + LLM）时，**哪个组件更敏感、该用什么方法、分配多少比特**？

实验在 BLIP-2 和 LLaVA 两种架构上进行，覆盖 captioning（COCO）、retrieval（Flickr）、VQA（VQAv2、GQA）四类任务，对比 uniform quantization、[[GPTQ]]、[[AWQ]] 三种方法。通过 Random Forest + Permutation + SHAP 三种归因方法的共识排名，量化了各组件的相对重要性。

## 五大关键发现

### 1. 组件敏感度与参数量不成正比

ViT 在 LLaVA 中仅占 **4.3%** 的参数量，但在 GPTQ 下对 VQA 任务的重要性达到 **20-28%**。这意味着不能简单按参数量分配比特预算——小组件也可能对最终性能至关重要。

### 2. SOTA 方法将"免费午餐"区域从 6-10 bpw 压缩到 3.5-4.5 bpw

| 方法 | 性能保持的 bpw 范围 |
|------|-------------------|
| Uniform Quantization | 6.0-8.0（retrieval）/ 8.0-10.0（captioning） |
| **GPTQ / AWQ** | **3.5-4.5**（所有任务） |

GPTQ 和 AWQ 在多模态场景下同样有效，这验证了 LLM 量化技术向多模态迁移的可行性。

### 3. 任务特性决定最优比特分配

| 任务类型 | LLM 重要性 | ViT 重要性 | 规律 |
|---------|-----------|-----------|------|
| **推理型 VQA**（VQAv2、GQA） | 73-98% | 2-28% | 强烈偏向 LLM 精度 |
| **Captioning**（COCO） | 48-81% | 17-50% | 较均衡 |
| **Retrieval**（Flickr） | 无 LLM | ViT 70-85% | ViT 主导 |

推理密集型任务（VQA）几乎完全依赖 LLM 精度，而视觉-文本对齐任务（captioning）对 ViT 和 LLM 有更均衡的依赖。

### 4. 量化方法本身改变组件重要性分布

这是最意想不到的发现——**同一模型、同一任务，换一种量化方法，组件的重要性排名可能完全不同**：

| 量化方法 | COCO Captioning (BLIP-2) | VQAv2 (LLaVA) |
|---------|--------------------------|----------------|
| **GPTQ** | ViT 50.4%, LLM 47.6% | ViT 27.9%, LLM 72.2% |
| **AWQ** | ViT 17.1%, LLM 80.5% | ViT 6.1%, LLM 93.9% |

AWQ 的激活感知策略优先保护 LLM 的大权重矩阵，导致 LLM 重要性被集中放大；GPTQ 的 Hessian 方法则能捕捉更广泛的跨组件交互，重要性分布更均匀。

**对部署的指导意义**：如果用 AWQ，应当给 LLM 更多比特预算；如果用 GPTQ，ViT 和 LLM 的比特分配应更均衡。

### 5. 组件间存在非加性交互效应

同时量化 ViT + LLM 的性能下降**大于**分别量化 ViT 和 LLM 的下降之和。这种非加性效应来自 pipeline 中的顺序依赖——ViT 的量化误差经 Connector 传播到 LLM，与 LLM 自身的量化误差叠加放大。

这意味着不能独立评估各组件的量化敏感度，需要**整体 pipeline 分析**。

## 前置实验：Block Groups 与 Layer Types 的影响

上述五大发现均聚焦于 **Model Components**（ViT / Q-Former / LLM）维度。实际上，论文在 Section 3.1 中还沿另外两个轴做了细粒度消融实验（图 2/3），但结论是它们的影响远小于组件级别，因此后续 SOTA 分析和重要性归因均聚焦于组件维度。

### Block Groups（前 / 中 / 后段）：位置不敏感

作者将每个模型组件均匀划分为 3 个连续块组（Front、Middle、End），测试了所有 $C_3^1 + C_3^2 + C_3^3 = 7$ 种组合。从图 3（右侧 panel）可见：

- 不同块组组合的性能-bpw 曲线**几乎重合**，没有观察到明显的分层规律
- 没有证据表明"前段"或"后段"等特定位置比其余部分更需要保护高精度
- 该结论在 captioning 和 retrieval 任务上均成立

**实用建议**：不应针对特定块组保留精度，而应统一处理所有块。这与 LLM 量化中"不同层敏感度不同"的常见结论不同——在 VLM 的 uniform quantization 设定下，块位置的影响被组件级差异所淹没。

### Layer Types（Attention vs FF）：必须同时量化

从图 3（中间 panel）可见三种配置：

- 只量化 Attention 层
- 只量化 FF（前馈）层
- 同时量化两种层（Both）

结果表明：单独量化某一种层类型的点**落在 Pareto 前沿下方**——只有同时量化 Attention 和 FF 才能达到最优的性能-体积权衡。

**实用建议**：不要选择性保护 Attention 层或 FF 层，两者都需要被量化才能达到最佳的 bpw-性能 trade-off。

### 为什么后续只关注 Components？

论文在第 4 页明确说明：

> "These findings are largely consistent on the LLM-free retrieval task as well, as we show in the appendix. Therefore, we focus the majority of our analysis in this section on the model components."

层类型和块组的结果在所有任务上都稳定且一致，说明 VLM 中**组件级别的敏感度差异远大于层内细分**。因此资源分配到组件级别（ViT vs LLM vs Q-Former）带来的优化空间远大于层内的块组或层类型级别。

## 组件重要性分析框架

本文提出了一个通用的量化敏感度分析框架，基于三种互补的归因方法：

| 方法 | 原理 | 特点 |
|------|------|------|
| **Random Forest Feature Importance** | 训练 RF 回归器预测量化后性能，提取特征重要性 | 捕捉非线性关系 |
| **Permutation Feature Importance** | 随机打乱某组件的比特值，观察性能下降 | 模型无关，直接度量 |
| **SHAP (Shapley Values)** | 博弈论方法，考虑所有特征组合计算贡献 | 实例级归因，最理论完备 |
| **Consensus Ranking** | 归一化后平均三种方法的结果 | 抵消单一方法偏差 |

线性模型对量化敏感度的拟合非常差（$R^2 < 0.20$），表明量化对多模态模型性能的影响是**高度非线性**的。

## Consensus 重要性表（完整数据）

### BLIP-2

| 方法 | 数据集 | ViT | Q-Former | LLM |
|------|--------|-----|----------|-----|
| GPTQ | COCO | 50.4% | 2.0% | 47.6% |
| GPTQ | GQA | 22.6% | 0.7% | 76.7% |
| GPTQ | VQAv2 | 25.3% | 1.5% | 73.1% |
| GPTQ | Flickr-Text | 70.3% | 29.7% | — |
| GPTQ | Flickr-Image | 78.4% | 21.6% | — |
| AWQ | COCO | 17.1% | 2.5% | 80.5% |
| AWQ | GQA | 1.0% | 0.4% | 98.6% |
| AWQ | VQAv2 | 1.5% | 0.2% | 98.3% |
| AWQ | Flickr-Text | 73.2% | 26.8% | — |
| AWQ | Flickr-Image | 84.7% | 15.3% | — |

### LLaVA

| 方法 | 数据集 | ViT | LLM |
|------|--------|-----|-----|
| GPTQ | GQA | 20.87% | 79.13% |
| GPTQ | VQAv2 | 27.85% | 72.15% |
| AWQ | GQA | 5.38% | 94.62% |
| AWQ | VQAv2 | 6.11% | 93.89% |

## Q-Former 的特殊角色

Q-Former（BLIP-2 的视觉-语言连接器）在大多数任务中重要性极低（<3%），但在**不使用 LLM 的 retrieval 任务**中重要性跳升至 **15-30%**。这揭示了一个有趣的动态：当 LLM 不存在时，Q-Former 必须独自承担视觉-文本对齐的全部责任；当 LLM 存在时，LLM 的强大语言能力可以补偿 Q-Former 的对齐不足。

LLaVA 使用简单的线性投影层替代 Q-Former，这使得 ViT 的相对重要性更高（因为投影层容量有限，更多视觉处理压力落在 ViT 上）。

## 与 LLM 量化的关系

本文直接使用 [[GPTQ]] 和 [[AWQ]] 作为 SOTA 方法，验证了 LLM 量化技术在多模态场景的迁移性：

| 维度 | 纯 LLM 量化 | VLM 量化（本文） |
|------|-----------|-----------------|
| 量化目标 | 单一 Transformer | ViT + Connector + LLM pipeline |
| 方法选择 | GPTQ / AWQ 直接应用 | 同样方法，但需考虑**跨组件**分配 |
| 敏感度分析 | per-layer / per-block | per-component（ViT vs LLM vs Connector） |
| 关键发现 | [[Emergent Outlier Features]] 驱动量化难度 | **任务类型**和**量化方法**共同决定组件重要性 |
| 最优 bpw | W4A16 / W4A8 为主流 | 3.5-4.5 bpw（pipeline 整体） |

### 本文未涉及但值得探索的方向

- 未测试 Rotation-based 方法（[[QuaRot]]、[[SpinQuant]]）在 VLM 上的效果
- 未探索激活量化（仅做 weight-only quantization）
- 未涉及视频 VLM（VideoChat、Video-LLaVA 等）
- 未与扩散模型量化（[[Diffusion Model Quantization]]）建立联系——后者也面临类似的多组件量化问题

## 对部署的实践指导

基于本文的发现，VLM 部署时的量化策略可归纳为：

1. **推理型任务（VQA）**：LLM 保持 8-bit 或更高，ViT 可激进量化到 4-bit
2. **生成型任务（Captioning）**：ViT 和 LLM 均衡分配，均用 4-bit GPTQ
3. **检索型任务（Retrieval）**：ViT 精度优先，Connector 也需保持较高精度
4. **方法选择**：AWQ 更适合 LLM-dominant 的任务；GPTQ 在需要均衡各组件的任务上更好
5. **整体策略**：3.5-4.5 bpw 是"免费午餐"边界，低于此质量快速下降

## 引用关系

← 使用的量化方法：[[GPTQ]]（Hessian-based weight quantization）、[[AWQ]]（activation-aware weight quantization）
← 相关概念：[[Weight-Only Quantization]]（本文所有实验均为 weight-only）、[[Emergent Outlier Features]]（LLM 组件的量化难度根源）
→ 扩展方向：[[Diffusion Model Quantization]]（类似的多组件量化问题——UNet/DiT 中不同子模块的敏感度差异）
→ 方法互补：[[Q-VLM]]（NeurIPS 2024）从方法角度解决 VLM 量化——用跨层依赖挖掘建模本文发现的"非加性交互效应"，在 LLaVA W4A4 上显著优于 AWQ/QLoRA
→ 应用场景：VLM 部署、边缘设备推理、多模态模型压缩
