---
title: "A Simple and Effective Pruning Approach for Large Language Models"
aliases: [Wanda, Sun 2024]
source_type: paper
source_url: "https://arxiv.org/abs/2306.11695"
source_date: 2023-06-20
ingested: 2026-04-28
venue: ICLR 2024
authors: [Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter]
affiliation: Carnegie Mellon University (CMU)
code: "https://github.com/locuslab/wanda"
tags: [model-compression, pruning, unstructured-sparsity, weight-pruning, activation-aware, inference-efficiency]
---

## 核心要点

Wanda（**W**eights **and** **a**ctivations）提出了一种极其简单但有效的 LLM 剪枝标准：**权重绝对值 * 输入激活范数**（$|W_{ij}| \cdot \|X_j\|_2$）。这一标准无需计算 Hessian 矩阵或进行迭代权重更新，仅需一次前向传播即可完成剪枝，运行速度比 [[SparseGPT]] 快一个数量级以上，同时在多数设置下取得了可比甚至更优的剪枝质量。

Wanda 的核心 insight 源自对 LLM 中 **Emergent Outlier Features**（涌现异常特征）的观察：某些输入特征维度的激活值异常大，通过这些维度的权重即使绝对值不大，对输出的实际影响也远大于其 magnitude 所暗示的。因此，剪枝时必须同时考虑权重和激活的联合重要性。

## 方法详解

### 剪枝标准

Wanda 的剪枝 metric 定义为：

$$S_{ij} = |W_{ij}| \cdot \|X_j\|_2$$

其中：
- $W_{ij}$ 是权重矩阵中第 $i$ 行第 $j$ 列的元素
- $\|X_j\|_2$ 是第 $j$ 个输入特征在校准数据上的 L2 范数
- 按每行（per-output）排序，剪掉 $S_{ij}$ 最小的权重

### 与 SparseGPT 的对比

| 特性 | SparseGPT | Wanda |
|------|-----------|-------|
| 需要 Hessian | 是 | **否** |
| 权重更新 | 迭代更新未剪枝权重 | **不更新** |
| 剪枝标准 | OBS-based | Weight * Activation |
| 运行时间 (LLaMA-7B) | ~30 min | **~1 min** |
| 50% 稀疏 PPL | 略优 | 可比 |

### 为什么如此简单的方法有效？

1. **Outlier features 的关键作用**：LLM 中约 1-7% 的特征维度具有异常大的激活值（参见 [[Emergent Outlier Features]]）。这些维度对应的权重在剪枝时必须被保护，而纯 magnitude pruning 会错误地剪掉它们。
2. **Per-row 比较的重要性**：在每行内独立排序确保了每个输出维度都保留了最重要的输入连接。

## 实验结果

| 模型 | 方法 | 50% 稀疏 PPL | 2:4 稀疏 PPL |
|------|------|-------------|-------------|
| LLaMA-7B | Magnitude | 54.6 | 78.3 |
| LLaMA-7B | SparseGPT | 8.22 | 10.87 |
| LLaMA-7B | **Wanda** | **8.27** | **11.12** |
| LLaMA-13B | Magnitude | 21.3 | 30.1 |
| LLaMA-13B | SparseGPT | 6.80 | 8.32 |
| LLaMA-13B | **Wanda** | **6.90** | **8.56** |

- 在 50% 非结构化稀疏度下，Wanda 与 SparseGPT 的 PPL 差距不到 0.1
- 在 2:4 半结构化稀疏格式下差距略大，但仍远优于 magnitude pruning
- 运行速度快 **30x+**

## 局限性

1. **不进行误差补偿**：不更新未剪枝权重意味着放弃了 Hessian-based 的误差补偿机制，在高稀疏度（>60%）下劣于 SparseGPT。
2. **依赖 outlier 特征假设**：如果未来模型（如使用了 [[SmoothQuant]] 式平滑训练的模型）不再具有明显的 outlier features，Wanda 的优势可能减弱。
3. **仅限于非结构化/半结构化稀疏**：对于结构化剪枝（如整行/整列移除），需要修改。

## 与已有知识的关联

- **与 [[SparseGPT]] 的关系**：Wanda 可视为 SparseGPT 的极简版本 -- 移除了 Hessian 计算和迭代权重更新，仅保留了"考虑输入激活"的核心 insight。
- **与 [[Emergent Outlier Features]] 的关系**：Wanda 的有效性直接源于 LLM 中 outlier 特征的存在。$\|X_j\|_2$ 项本质上是在度量每个输入特征的"异常程度"。
- **与 [[SmoothQuant]] 的关系**：SmoothQuant 试图消除 outlier 以便量化，而 Wanda 利用 outlier 来指导剪枝，两者视角互补。
- **与 [[GPTQ]]、[[AWQ]] 的关系**：同属后训练压缩方法，GPTQ/AWQ 面向量化，Wanda 面向剪枝。
- **与 [[Pruning and Sparsity]] 的关系**：Wanda 证明了在 LLM 时代，简单方法配合正确的 insight 可以匹敌复杂的优化方法。

## 引用关系

### 被引用 ->
- -> 大量后续 LLM 剪枝工作将 Wanda 作为标准 baseline

### 引用的前置工作 <-
- <- [[SparseGPT]]：Hessian-based one-shot pruning（主要对比方法）
- <- [[GPTQ]]：共享逐层处理的设计思路
- <- Emergent outlier 研究 (Dettmers et al., 2022)
