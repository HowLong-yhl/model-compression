---
title: "OSTQuant: Refining LLM Quantization with Orthogonal and Scaling Transformations"
aliases: [OSTQuant]
tags: [model-compression, quantization, source, rotation, scaling, PTQ, optimized-rotation, optimized-scaling, W4A4]
created: 2026-04-20
updated: 2026-04-20
type: source-summary
venue: arXiv 2025 (under review)
authors: [Xing Hu, Yuan Cheng, Dawei Yang, Zukang Xu, Zhihang Yuan, Jiangyong Yu, Chen Xu, Zhixuan Chen, Zhe Jiang, Sifan Zhou]
affiliation: Houmo AI / Nanjing University / Southeast University
url: https://arxiv.org/abs/2501.13987
code: https://github.com/BrotherHappy/OSTQuant
---

## 一句话总结

OSTQuant 提出 **QSUR（Quantization Space Utilization Rate）** 指标，从理论上证明了正交变换+缩放变换的组合能达到量化空间利用率的最大化，并设计了一套全局优化框架用 **Riemann Adam** 在 Stiefel 流形上联合学习旋转矩阵 O 和缩放矩阵 Λ。

## 核心问题

现有方法（SmoothQuant 做 scaling，QuaRot 做 rotation）各有局限——缩放对 outlier 敏感、旋转对通道方差不敏感。**如何证明两者组合的必要性，并端到端优化？**

## 理论贡献：QSUR 指标

### 定义

给定 d 维数据 X，QSUR 定义为数据实际占据的超体积 V_X 与量化空间超体积 V_S 的比值：

```
QSUR_X = V_X / V_S
```

QSUR 越高，说明数据分布越"填满"量化空间，量化精度越高。

### 理论推导

对服从 N(µ, Σ) 的数据，Σ = QΛQ⊤（特征分解），QSUR 与以下因素正相关：

- **缩放变换 Λ^{-1/2}**：消除特征值不均（inter-channel variance），提升 QSUR
- **正交变换 Q⊤**：消除特征向量方向的 outlier，进一步提升 QSUR
- **组合 T = c·Λ^{-1/2}·Q⊤**：达到理论最大 QSUR

**核心定理**：单独 Scaling 或单独 Rotation 均无法达到最大 QSUR；只有两者组合才能最大化量化空间利用率。

## 方法详解

### 变换对（Transformation Pair）

每个变换对定义为 `T = Λ · O`，其中：
- O ∈ R^{n×n}：可学习正交矩阵（在 Stiefel 流形上）
- Λ ∈ R^{n×n}：可学习对角缩放矩阵

前向传播变为：
```
y = Q(x · W₁ · O · Λ) · Q(Λ⁻¹ · O⊤ · W₂)
```

变换后 `Λ⁻¹ · O⊤ · W₂` 可预计算融入权重，部署时零开销。

### 多层级变换结构

每个 Transformer block 学习 4 个变换对：
- **R_res**：全局残差流旋转（跨 block 传播）
- **V-O pair**：注意力值投影与输出投影之间
- **Up-Down pair**：FFN 上投影与下投影之间
- **Gate pair**：FFN gate 与 down 之间

### 优化策略

| 组件 | 优化方法 | 约束 |
|------|---------|------|
| 正交矩阵 O | **Riemann Adam** | Stiefel 流形（O⊤O = I） |
| 缩放矩阵 Λ | 标准 Adam | 对角正定（直接学习对角元素） |
| 初始化（WOMI） | 权重协方差特征分解 + 归一化 Hadamard | 比随机初始化收敛更快 |

### KL-Top Loss

创新损失函数：仅对全精度模型输出的 **top-k logits** 计算 KL 散度，避免长尾 token 的噪声干扰：

```
L_KL-Top = KL(softmax(ŷ_topk) || softmax(y_topk))
```

## 实验亮点

| 模型 | 设定 | OSTQuant | vs QuaRot | vs SpinQuant |
|------|------|----------|-----------|--------------|
| LLaMA-3 8B | W4A4KV4 | 保留 96%+ 性能 | 显著提升 | 降低 32% 性能差距 |
| LLaMA-3 8B | W4-only | >99.5% 保留 | — | — |
| 量化速度 | A800 GPU | **20 分钟** | 更快 | 可比 |

## 与其他方法的关系

- **vs [[QuaRot]]**：QuaRot 用随机旋转（不优化），OSTQuant 用 Riemann Adam 学习旋转，QSUR 理论解释了为什么学习更好
- **vs [[SpinQuant]]**：两者都优化旋转，但 SpinQuant 仅优化旋转，OSTQuant **联合优化旋转+缩放**，QSUR 证明了缩放不可或缺
- **vs [[SmoothQuant]] / [[AWQ]]**：这些是未优化的 scaling，OSTQuant 的 learnable Λ 通过梯度下降更精确
- **vs [[FlatQuant]]**：同期竞争方法，都做 learned rotation + scaling，但 FlatQuant 用 Kronecker 分解，OSTQuant 用全尺寸正交矩阵

## 局限性

- 全尺寸正交矩阵在推理时需要在线计算（vs FlatQuant 的 Kronecker 分解更快）
- 理论假设数据为高斯分布（实际 LLM 激活可能有偏）
- 论文尚在审稿中（arXiv preprint）

## 引用关系

← 理论基础：Stiefel 流形优化 (Riemann Adam), [[SpinQuant]] (Cayley SGD 对比), [[QuaRot]] (Computational Invariance)
← 分类：[[Optimized Rotation]] + [[Optimized Scaling]]（联合优化正交旋转与对角缩放）
← 对比基线：[[SmoothQuant]], [[AWQ]], [[QuIP#]], QuaRot, SpinQuant
→ 理论贡献：QSUR 指标为后续优化方法提供了统一评价标准
→ 相关：[[Quantization Insights]]（Insight 1 的理论基础）、[[Equivalent Scaling Transformation]]
