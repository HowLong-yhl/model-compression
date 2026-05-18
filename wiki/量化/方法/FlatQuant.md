---
title: "FlatQuant: Flatness Matters for LLM Quantization"
aliases: [FlatQuant, FLATQUANT]
tags: [model-compression, quantization, source, rotation, scaling, PTQ, optimized-rotation, optimized-scaling, W4A4]
created: 2026-04-20
updated: 2026-04-20
type: source-summary
venue: ICML 2025
authors: [Yuxuan Sun, Ruikang Liu, Haoli Bai, Han Bao, Kang Zhao, Yuening Li, Jiaxin Hu, Xianzhi Yu, Lu Hou, Chun Yuan, Xin Jiang, Wulong Liu, Jun Yao]
affiliation: Huawei Noah's Ark Lab / Tsinghua University / CUHK
url: https://arxiv.org/abs/2410.09426
code: https://github.com/ruikangliu/FlatQuant
---

## 一句话总结

FlatQuant 提出 **flatness（平坦度）** 是量化的关键目标，通过 **Kronecker 分解的可学习仿射变换** 实现对每个线性层的最优预量化旋转，结合 learnable scaling 和 clipping，首次在 LLaMA-3-70B 上实现 ≤1% accuracy drop 的 W4A4 量化（仅用 RTN）。

## 核心问题

Per-channel scaling 和固定 Hadamard 旋转虽能部分消除 outlier，但仍留下**不平坦的分布**（steep envelopes、残余 outlier），导致量化误差在 Transformer 层间累积。如何实现**真正平坦**的权重和激活分布？

## 核心洞察：Flatness

"Flat" 分布的含义：排序后的通道幅度曲线接近水平线（低 kurtosis），使得量化步长 Δ 能均匀覆盖所有值。FlatQuant 观察到：

1. **Per-channel scaling** 能平坦化激活，但代价是权重变得更陡峭
2. **Hadamard 旋转** 两者都能改善，但在某些层仍不够平坦
3. **可学习的逐层仿射变换** 可以同时使权重和激活都平坦

## 方法详解

### 可学习仿射变换 P（Learned Affine Transformation）

对线性层 Y = XW⊤，找最优可逆矩阵 P：

```
P* = argmin_P ‖Y - Q(XP) · Q(P⁻¹W⊤)‖²_F
```

变换后权重 P⁻¹W⊤ 可离线预计算融入权重。

### Kronecker 分解（核心创新）

全尺寸 P ∈ R^{n×n} 太大（2× 计算开销）。FlatQuant 将 P 分解为 Kronecker 积：

```
P = P₁ ⊗ P₂,   P₁ ∈ R^{n₁×n₁}, P₂ ∈ R^{n₂×n₂}, n₁·n₂ = n
```

**效率优势**：
- 存储减少 n/2 倍（最优时 n₁ = n₂ = √n）
- 计算节省 √n/2 倍
- 实际开销仅占总 FLOPs 的 **~2.61%**，额外内存 **3.41 MB**（LLaMA-2-7B）
- 最优配置：n=8192 时取 (n₁, n₂) = (64, 128)

### 三组联合可学习参数 Θ

| 参数 | 类型 | 作用 | 融合方式 |
|------|------|------|---------|
| P = P₁ ⊗ P₂ | 可学习仿射变换 | 旋转/重分配通道 | 在线计算（Kronecker 乘法） |
| diag(c) | 可学习 per-channel 缩放 | 平衡 outlier | 融入前一层 LayerNorm（零开销） |
| α_w, α_a | 可学习 clipping 阈值 | sigmoid 参数化 | 融入量化过程（零开销） |

### 优化目标

Block-wise MSE 最小化（128 sentences calibration）：

```
min_Θ ‖F_l(X) - F̂_l(X; Θ)‖²_F
```

其中 Θ = {P, c, α_a, α_w}，逐 block 优化。

### 高效推理 Kernel

所有操作（Kronecker 仿射变换 + 量化）融合为**单个 Triton kernel**：
- 中间结果保存在 SRAM，消除 redundant global memory access
- 减少 kernel launch overhead

## 实验亮点

| 模型 | 设定 | FlatQuant | vs SpinQuant | vs QuaRot |
|------|------|-----------|--------------|-----------|
| LLaMA-3 70B | W4A4 | **≤1% drop** | +7.5 pp | 大幅领先 |
| LLaMA-3 8B | W4A4 | SOTA | 显著优于 | 显著优于 |

- **首次**在 LLaMA-3-70B 上用纯 RTN 实现 ≤1% accuracy drop 的 W4A4
- Prefill **2.3× 加速**，Decoding **1.7× 加速**（vs FP16 baseline）
- 全部使用 RTN（Round-to-Nearest），无需 GPTQ

## 与其他方法的关系

- **vs [[QuaRot]]**：QuaRot 用固定 Hadamard（同一矩阵所有层共享），FlatQuant 为每层学习独立的最优仿射变换
- **vs [[SpinQuant]]**：SpinQuant 学习全局旋转（在 Stiefel 流形上），FlatQuant 学习逐层仿射（通过 Kronecker 分解），同时还学习 scaling 和 clipping
- **vs [[OSTQuant]]**：同期竞争方法，两者都联合优化旋转+缩放。OSTQuant 用全尺寸正交矩阵 + Riemann Adam；FlatQuant 用 Kronecker 分解（更高效）+ 标准梯度（非流形约束）
- **vs [[SmoothQuant]] / [[AWQ]]**：这些仅做 scaling 且不优化；FlatQuant 同时做 learned rotation + learned scaling + learned clipping
- **核心差异**：FlatQuant 的 P 不严格约束为正交矩阵（是一般可逆矩阵），而 SpinQuant/OSTQuant 严格约束正交性

## 局限性

- Kronecker 仿射变换仍需在线计算（~2.6% FLOPs 开销）
- 每层独立的 P 矩阵增加了少量模型存储
- P 非正交约束，理论保证弱于 SpinQuant/OSTQuant（但实践效果更好）
- 需要自定义 Triton kernel 才能实现高效推理

## 引用关系

← 启发：Per-channel scaling (SmoothQuant), Hadamard rotation (QuaRot), Learnable transforms (OmniQuant)
← 分类：[[Optimized Rotation]]（可学习仿射变换，不严格正交）+ [[Optimized Scaling]]（learnable diag(s)）
← 对比基线：[[QuaRot]], [[SpinQuant]], OmniQuant, [[AWQ]]
→ 贡献：证明 flatness 是量化的关键指标，Kronecker 分解使逐层可学习变换在实践中可行
→ 相关：[[Quantization Insights]]、[[Equivalent Scaling Transformation]]
