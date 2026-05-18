---
title: "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot"
aliases: [SparseGPT, Frantar 2023]
source_type: paper
source_url: "https://arxiv.org/abs/2301.00774"
source_date: 2023-01-02
ingested: 2026-04-28
venue: ICML 2023
authors: [Elias Frantar, Dan Alistarh]
affiliation: IST Austria (Institute of Science and Technology Austria)
code: "https://github.com/IST-DASLab/sparsegpt"
tags: [model-compression, pruning, unstructured-sparsity, one-shot, hessian, weight-pruning, inference-efficiency]
---

## 核心要点

SparseGPT 是首个在不进行任何重训练的情况下，将 **100B+ 参数的 LLM** 准确剪枝到 **50-60% 稀疏度** 的方法。其核心思想是将权重剪枝问题转化为 **逐层的稀疏回归问题**，利用近似 Hessian 信息（基于少量校准数据计算的 $X X^\top$ 矩阵）来最小化剪枝引起的输出误差。SparseGPT 的算法框架与同一团队的权重量化方法 [[GPTQ]] 高度相关，两者共享 Hessian-based 逐列更新的核心思想。

在 OPT-175B 和 BLOOM-176B 上，SparseGPT 在 50% 非结构化稀疏度下仅导致极小的 perplexity 增加（通常 <1 point），且整个剪枝过程可在单 GPU 上在数小时内完成。结合 2:4 半结构化稀疏格式，还可利用 NVIDIA A100 的硬件稀疏加速获得实际推理加速。

## 方法详解

### 逐层剪枝框架

SparseGPT 对每一层独立执行以下步骤：

1. **收集校准数据**：使用少量校准样本（~128 条）通过前向传播收集每层的输入激活 $X$
2. **计算近似 Hessian**：$H = 2 X X^\top$（二阶信息的近似）
3. **逐列更新**：按列遍历权重矩阵，对每列：
   - 选择要剪枝的权重（magnitude 最小的）
   - 将剪枝误差通过 Hessian 信息传播到后续列的未剪枝权重上（误差补偿）
   - 更新后续列的权重以最小化总输出误差

这一过程本质上是 Optimal Brain Surgeon (OBS) 的高效近似版本，将 $O(d^3)$ 的复杂度降低到实际可行的水平。

### 与 GPTQ 的关系

SparseGPT 与 [[GPTQ]] 共享同一个 **逐层 Hessian-based 优化框架**：
- **GPTQ**：将每列的权重 **量化** 到低比特，用 Hessian 补偿量化误差
- **SparseGPT**：将每列的部分权重 **置零**，用 Hessian 补偿剪枝误差
- 两者可以 **联合使用**：先稀疏再量化，或在同一框架内同时执行

## 实验结果

| 模型 | 稀疏度 | 原始 PPL | SparseGPT PPL | 差异 |
|------|--------|---------|--------------|------|
| OPT-175B | 50% (unstructured) | 8.34 | 8.68 | +0.34 |
| OPT-175B | 2:4 semi-structured | 8.34 | 9.13 | +0.79 |
| OPT-66B | 50% unstructured | 9.34 | 9.72 | +0.38 |
| BLOOM-176B | 50% unstructured | ~8.1 | ~8.5 | +0.4 |

- 50% 非结构化稀疏度下，perplexity 增加不到 0.5
- 60% 稀疏度仍可保持合理质量（PPL 增加 <2）
- 剪枝 OPT-175B 仅需约 4 小时（单 A100 GPU）

## 局限性

1. **非结构化稀疏缺乏硬件加速**：50% 非结构化稀疏在通用 GPU 上难以获得实际加速，需要 2:4 格式或专用硬件。
2. **校准数据依赖**：Hessian 的质量取决于校准数据的代表性。
3. **大稀疏度下质量显著下降**：超过 60% 稀疏度时，质量损失加速增大。

## 与已有知识的关联

- **与 [[GPTQ]] 的关系**：同一框架（Hessian-based 逐层优化），GPTQ 用于量化，SparseGPT 用于剪枝。两者可组合使用。
- **与 [[Wanda]] 的关系**：Wanda 是 SparseGPT 的极简替代 -- 不使用 Hessian 信息，仅用 weight * activation magnitude 作为剪枝标准，速度更快但精度稍低。
- **与 [[AWQ]] 的关系**：AWQ 也使用激活信息指导权重处理，但目标是量化而非剪枝。
- **与 [[Pruning and Sparsity]] 的关系**：SparseGPT 是 LLM 时代 one-shot pruning 的开创性工作。

## 引用关系

### 被引用 ->
- -> [[Wanda]]：受 SparseGPT 启发的简化剪枝方法
- -> [[GPTQ]]：共享 Hessian-based 框架（同一团队）

### 引用的前置工作 <-
- <- Optimal Brain Surgeon (LeCun et al., 1989; Hassibi & Stork, 1993)
- <- [[GPTQ]] / OPTQ (Frantar et al., 2022)：同一团队的量化工作
