---
title: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
aliases: [TurboQuant, Zandieh 2024]
source_type: paper
source_url: "https://arxiv.org/abs/2502.02754"
source_date: 2025-02-04
ingested: 2026-04-28
venue: ICML 2025
authors: [Amir Zandieh, Insu Han, Majid Daliri, Amin Karbasi]
affiliation: Google Research / Google DeepMind
code: null
tags: [model-compression, quantization, method, kv-cache, vector-quantization, theory, data-oblivious, online, rotation]
---

## 核心要点

TurboQuant 从信息论角度提出了一种**近最优的在线向量量化（Online VQ）**方法，核心思路是：对输入向量施加 random rotation 后，各坐标近似独立并服从 Beta 分布，从而可以对每个坐标独立应用 Lloyd-Max 最优标量量化器。该方法在理论上达到 Shannon 蒸馏率下界的 ~2.7× 以内，且完全 **data-oblivious**（不需要 calibration 数据），可直接用于 KV cache 的在线量化。

与 [[KVQuant]] 和 [[KIVI]] 等启发式方法不同，TurboQuant 从理论出发建立了 KV cache 量化的蒸馏率下界，并设计了可证明接近最优的量化器。这也是首次将随机旋转的理论保证（从 incoherence/JL 引理的角度）与量化质量的 rate-distortion 理论统一起来。

## 方法详解

### 核心思路：Random Rotation → Beta Distribution → Optimal Scalar Quantizer

**Step 1：Random Rotation**

对输入向量 $x \in \mathbb{R}^d$ 施加随机正交旋转 $R$（或随机 Hadamard 矩阵），得到 $\tilde{x} = Rx$。

**关键定理**：对于 $\|x\| = 1$ 的单位向量，random rotation 后每个坐标 $\tilde{x}_i^2$ 服从 Beta(1/2, (d-1)/2) 分布。当维度 d 较大时（LLM 中 d=128 per head），该 Beta 分布近似 Gaussian $\mathcal{N}(0, 1/d)$。

**核心洞察**：rotation 后各坐标近似独立且同分布——这意味着可以对每个坐标独立应用相同的标量量化器，而无需了解原始数据分布（data-oblivious）。

**Step 2：Per-coordinate Optimal Scalar Quantization**

对 Beta(1/2, (d-1)/2) 分布，使用 Lloyd-Max 算法（最优均方误差标量量化器）预计算 signpost。由于分布已知且固定，signpost 可一次性离线计算。

**与 uniform quantization 的对比**：Beta/Gaussian 分布中心密集、尾部稀疏，Lloyd-Max 在中心区域放置更密的 signpost，比 uniform 量化更高效。

**Step 3：Norm 独立处理**

将 $x$ 分解为方向和 norm：$x = \|x\| \cdot \hat{x}$。Norm $\|x\|$ 以高精度单独存储（通常 16-bit），方向 $\hat{x}$ 经旋转后逐坐标量化。

### 用于内积的两阶段量化

KV cache 的核心操作是 $QK^T$（内积），TurboQuant 为内积场景设计了两阶段方案：

**Stage 1：MSE 量化器（b-1 bits/coordinate）**

对旋转后的向量用 b-1 bit 的 Lloyd-Max 量化器，得到量化值 $\hat{x}$ 和残差 $r = \tilde{x} - \hat{x}$。

**Stage 2：QJL 量化器（1 bit/coordinate）**

对残差 $r$ 应用 **Quantized Johnson-Lindenstrauss (QJL)** 变换——用 1-bit 随机投影（仅存符号位）近似残差的内积贡献。QJL 的理论保证：

$$\mathbb{E}[\langle \text{QJL}(r_x), \text{QJL}(r_y) \rangle] = \langle r_x, r_y \rangle$$

总比特预算为 b bits/coordinate：(b-1) bits MSE + 1 bit QJL。

### 理论保证

**定理（Distortion Rate Bound）**：TurboQuant 在 b bits/coordinate 下的 MSE 蒸馏率满足：

$$D(b) \leq C \cdot D^*(b)$$

其中 $D^*(b)$ 是 Shannon 理论下界，$C \approx 2.72$（即不超过最优值的 ~2.7 倍）。这个 bound 对**所有 bit-width** 都成立。

**对比**：传统 uniform quantization 在低比特下的蒸馏率可以是最优值的 $\Omega(d)$ 倍（随维度增长），TurboQuant 的 bound 与维度无关。

### 与旋转量化的联系

TurboQuant 的 random rotation 与 [[QuaRot]] / [[SpinQuant]] 的 Hadamard 旋转本质上解决同一问题——消除 outlier，使分布均匀化。区别在于：

- QuaRot/SpinQuant：旋转后用 uniform INT4 量化，目标是 weight+activation+KV 全量化
- TurboQuant：旋转后用 optimal non-uniform 量化，专注 KV cache，追求理论最优

## 实验结果

### KV Cache 量化

在 LLaMA-2-7B / 13B 上，KV cache 量化的 PPL 结果：

| 方法 | bits/dim | LLaMA-2-7B PPL | LLaMA-2-13B PPL |
|------|---------|----------------|-----------------|
| FP16 | 16 | 5.47 | 4.88 |
| KIVI 2-bit | 2 | 5.56 | 4.95 |
| KVQuant 3-bit | 3 | 5.74 | — |
| TurboQuant | 3.5 | ~5.50 | ~4.90 |
| TurboQuant | 2.5 | ~5.65 | ~5.00 |

TurboQuant 在 3.5 bits/dim 下实现了近无损量化（quality neutral），2.5 bits 下退化仍然很小。

### 理论 vs 实验

实验验证了理论 bound 的紧致性：实际 MSE 蒸馏率约为 Shannon 下界的 2.3-2.8×，与理论 bound（~2.72×）一致。

### Nearest Neighbor Search

TurboQuant 不仅适用于 KV cache，还在最近邻搜索（ANN）上有应用。在 GloVe-100 和 SIFT-128 数据集上，TurboQuant 在相同比特预算下的 recall 优于 Product Quantization（PQ）和 ScaNN。

## 局限性

- **计算开销**：random rotation（Hadamard 变换）需要 O(d log d) 额外计算，虽然快但非零
- **Norm 的额外存储**：norm 需要高精度存储（16-bit），在极低比特下占总存储的非零比例
- **非自适应**：data-oblivious 意味着不利用数据分布的特殊结构（如 Key 的固定 outlier channel），在某些场景下可能不如 data-aware 方法
- **缺乏完整的系统级 benchmark**：论文主要关注理论和 PPL 指标，缺乏端到端推理延迟和吞吐量的系统评测
- **仅做 MSE 优化**：量化目标是最小化 MSE，未考虑 task-aware 或 sensitivity-weighted 的目标（如 [[KVQuant]] 的 sensitivity-weighted k-means）

## 与已有知识的关联

### 与 Incoherence 理论的联系

TurboQuant 的 random rotation 从 [[Incoherence]] 理论的角度看非常自然——rotation 使矩阵的 incoherence 达到最小值（所有 entry 大小接近 $1/\sqrt{d}$），这正是 [[QuIP#]] 的 RHT 所追求的。TurboQuant 的贡献是将 incoherence → 量化质量的关系精确量化为 rate-distortion bound。

### 旋转 + 量化的统一视角

在本 wiki 的 [[Rotation-based Quantization]] 框架中，TurboQuant 可以被理解为"rotation + optimal scalar VQ"的组合：

```
QuaRot:      Random Hadamard → Uniform INT4 量化
SpinQuant:   Learned Rotation → Uniform INT4 量化
QuIP#:       Random Hadamard → E₈ 格码本 VQ
TurboQuant:  Random Rotation → Lloyd-Max 最优标量量化 + QJL 残差
```

TurboQuant 填补了"random rotation + non-uniform scalar quantization"这个组合的空白，并提供了理论最优性保证。

### 与 QJL 的关系

TurboQuant 的第二阶段使用了 QJL（Quantized Johnson-Lindenstrauss）变换。QJL 本身也是一个独立的 KV cache 量化方法（Zandieh et al., AAAI 2025），用 1-bit 随机投影压缩 Key cache 以保持内积。TurboQuant 将 QJL 从独立方法升级为两阶段框架中的残差量化器，充分利用了其在内积保持上的理论保证。

## 引用关系

← 理论基础：Shannon Rate-Distortion Theory，Johnson-Lindenstrauss Lemma，[[Incoherence]]（rotation 的理论根基）
← 量化技术：Lloyd-Max Optimal Quantizer，QJL (Zandieh et al., AAAI 2025)
← 相关方法：[[KVQuant]]（NUQ 的启发式版本），[[KIVI]]（per-channel/per-token 方向），Product Quantization
→ 贡献：首个有理论最优性保证的 KV cache 在线量化方法
→ 理论意义：建立了 rotation + scalar quantization 的 rate-distortion 统一框架
→ 概念关联：[[KV Cache Quantization]]（理论最优方案），[[Rotation-based Quantization]]（旋转在量化中的另一应用）
