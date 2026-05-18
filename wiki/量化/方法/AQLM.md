---
title: "AQLM: Extreme Compression of Large Language Models via Additive Quantization"
aliases: [AQLM, Additive Quantization for Language Models, Multi-Codebook Quantization]
source_type: paper
source_url: "https://arxiv.org/abs/2401.06118"
source_date: 2024-01-11
ingested: 2026-05-09
venue: ICML 2024
authors: [Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, Dan Alistarh]
affiliation: HSE University / Yandex Research / Skoltech / IST Austria / NeuralMagic
code: "https://github.com/Vahe1994/AQLM"
tags: [model-compression, quantization, method, post-training, weight-only, vector-quantization, extreme-compression, 2-bit]
---

## 核心要点

AQLM 将经典信息检索领域的 **Additive Quantization (AQ)** 方法扩展到 LLM 权重压缩，是首个在 2-3 bit 极限压缩区间达到 Pareto 最优的方案。与 GPTQ/AWQ 等"直接量化"（每个标量独立映射到量化网格）不同，AQLM 属于**向量量化（VQ）** 范式——将多个权重联合编码，利用权重间的互信息实现更高压缩率。

核心创新：(1) 将 AQ 的 MAP-MRF 优化问题改造为 **input-aware** 形式（最小化层输出误差而非权重重建误差）；(2) 引入 **intra-block joint fine-tuning**（跨层联合优化 codebook 参数），使量化误差在 transformer block 级别得到补偿。

## 方法摘要

### 向量量化编码

AQLM 将权重矩阵 W 的每一行分割为长度 g 的 group，每个 group 编码为 M 个 codebook 向量的**加法组合**：

```
W_hat[i,j] = sum_{m=1}^{M} C_m * b_{i,j,m}
```

其中 C_m ∈ R^{g × 2^B} 是第 m 个 codebook（含 2^B 个 g 维向量），b_{i,j,m} ∈ R^{2^B} 是 one-hot 编码索引。每个 group 使用 M·B 比特编码，有效比特宽度为 **M·B/g**（加 codebook 开销 g·2^B·16 bit 的 FP16 码本存储）。

典型配置：
- **2 bit**：g=8, M=1, B=16（即 1 个 codebook，每码 16 bit，每 8 权重 → 2 bpw）
- **2 bit**：g=8, M=2, B=8（即 2 个 codebook，每码 8 bit，每 8 权重 → 2 bpw）
- **3 bit**：g=8, M=2, B=12 或类似配置

与 Product Quantization（PQ，将向量**拼接**为子向量独立编码）不同，AQ 用**加法**组合多个 codebook 向量来逼近原始向量，不约束子空间独立，逼近精度更高。

### Input-Aware 优化目标

经典 AQ 最小化权重重建误差 ||W - Ŵ||²。AQLM 改为最小化**层输出误差**：

```
argmin_{C,b} ||WX - Ŵ X||²₂
```

其中 X ∈ R^{d_in × n} 为 calibration 数据的层输入。这使得量化误差直接以下游任务表现为导向——高激活方向上的权重误差被更严厉惩罚，低激活方向上允许更大误差。

关键数学简化：通过展开平方并预计算 **XX^T ∈ R^{d_in × d_in}**，避免存储 n 个 calibration sample 的完整激活，将空间复杂度从 O(d_in · n) 降至 O(d_in²)：

```
<A, B>_{XX^T} := <A·XX^T, B>_F
```

所有后续的 beam search 和 codebook 更新都基于此内积形式。

### 三阶段优化算法

AQLM 交替优化离散的 codes b 和连续的 codebooks C，直至收敛：

**Phase 1: Beam Search for Codes**

将 code 分配问题重构为 fully-connected discrete MRF 的 MAP 推断。目标函数（Eq. 7）分解为：
- **Unary potentials**：<W, C_m b_m>_{XX^T}（code 与原始权重的匹配度）
- **Pairwise potentials**：<C_i b_i, C_j b_j>_{XX^T}（不同 codebook codes 间的交互）

beam search 维护 k 个最优配置，每步尝试替换一个 code 并保留 top-k。各输出维度可**并行**处理。

**Phase 2: Codebook Update via Adam**

固定 codes b，最小化 ||(W - Ŵ)·XX^T·(W - Ŵ)^T||_F 关于 C_m 的梯度。使用 100 步 Adam (lr=10⁻⁴) 全批次梯度下降。虽然存在闭式解（需反转大矩阵），Adam 在实践中足够快且收敛稳定。同时学习 per-output-unit scales s_i := ||W_i||₂。

**Phase 3: Intra-Block Fine-Tuning**

前两阶段独立压缩每层，Phase 3 在 **transformer block 级别**联合优化：

```
min ||block(X_block) - Y_block||²₂
```

其中 Y_block 是量化前该 block 的原始输出。通过 PyTorch autograd 反向传播更新所有 codebooks C_m、scales s、以及非量化参数（RMSNorm scales/biases），codes b 保持冻结。这使不同层的量化误差能相互补偿——Phase 3 占总校准时间的 10-30%，但对 2-bit 精度贡献显著。

### 初始化：Residual K-Means

1. 对权重 groups 运行 K-means 聚类 → 第一个 codebook C₁
2. 计算量化残差 r = W_group - C₁[nearest]
3. 对残差运行 K-means → 第二个 codebook C₂
4. 以此类推初始化 M 个 codebooks

这保证每个后续 codebook 补偿前序 codebook 的量化残差。

## 实验结果

### 2-bit 压缩（核心优势区间）

| Model | Method | Avg bits | Wiki2 PPL↓ | C4 PPL↓ | Avg Zero-shot Acc↑ |
|-------|--------|----------|-----------|---------|-------------------|
| LLAMA2-7B | FP16 | 16 | 5.12 | 6.63 | 62.35% |
| LLAMA2-7B | QuIP# | 2.02 | 8.22 | 11.01 | 52.23% |
| LLAMA2-7B | **AQLM** | **2.02** | **6.59** | **8.54** | **57.28%** |
| LLAMA2-13B | FP16 | 16 | 4.57 | 6.05 | 65.38% |
| LLAMA2-13B | QuIP# | 2.01 | 6.06 | 8.07 | 57.55% |
| LLAMA2-13B | **AQLM** | **1.97** | **5.60** | **7.49** | **61.32%** |
| LLAMA2-70B | FP16 | 16 | 3.12 | 4.97 | 70.17% |
| LLAMA2-70B | QuIP# | 2.01 | 4.16 | 6.01 | 67.67% |
| LLAMA2-70B | **AQLM** | **2.07** | **3.94** | **5.72** | **68.75%** |

AQLM 在 2-bit 区间全面超越 QuIP#，尤其在 70B 模型上差距最为显著（PPL 3.94 vs 4.16）。

### 3-bit 压缩

| Model | Method | Wiki2 PPL↓ | C4 PPL↓ | Avg Acc↑ |
|-------|--------|-----------|---------|---------|
| LLAMA2-7B | GPTQ | 8.06 | 10.61 | 53.08% |
| LLAMA2-7B | SpQR | 6.20 | 8.20 | 59.07% |
| LLAMA2-7B | **AQLM** | **5.46** | **7.08** | **60.88%** |
| LLAMA2-13B | GPTQ | 5.85 | 7.86 | 59.61% |
| LLAMA2-13B | QuIP | 5.12 | 6.79 | 63.15% |
| LLAMA2-13B | **AQLM** | **4.82** | **6.37** | **64.49%** |
| LLAMA2-70B | GPTQ | 4.40 | 6.26 | 65.41% |
| LLAMA2-70B | SpQR | 3.85 | 5.63 | 68.22% |
| LLAMA2-70B | **AQLM** | **3.36** | **5.17** | **69.86%** |

3-bit AQLM 70B（3.36 PPL）已非常接近 FP16 baseline（3.12 PPL），差距仅 0.24。

### 推理速度

- **GPU**：~30% layer-wise 加速（vs FP16），得益于内存带宽减少
- **CPU**：最高 4× 加速
- 总体：2-bit AQLM 可 match 或超越 FP16 推理速度，同时内存占用仅 1/8

## 与两步分解框架的关系

AQLM **超越了两步分解框架**（Step 1 预量化变换 + Step 2 误差补偿），属于完全不同的量化范式：

| 维度 | 直接量化（GPTQ/AWQ/SpinQuant） | 向量量化（AQLM/QuIP#） |
|------|-------------------------------|---------------------|
| 编码单元 | 单个标量 → 网格点 | 多个权重联合 → codebook 向量 |
| 信息利用 | 仅单元自身值 | 组内权重的互信息 |
| 量化网格 | 固定（INT/FP/NF 格式） | **学习**的 codebook |
| 压缩极限 | ~3 bit（4-bit 为甜蜜点） | **2 bit 仍可用** |
| 计算 | 简单反量化 | 查表 + 加法（LUT-based） |
| 量化成本 | 分钟～小时 | **天**（beam search + fine-tuning） |

AQLM 的 input-aware 目标和 intra-block tuning 使其在极低比特率下获得远超直接量化方法的精度，但代价是数量级更高的 calibration 时间。

## 与 QuIP# 的对比

二者都是 2-bit 区间的前沿方法，但技术路线截然不同：

- **QuIP#**：旋转（RHT）使权重趋近高斯 → 固定 E₈ 格码本编码 → 最小化"舍入"误差
- **AQLM**：不旋转 → **学习** codebook 直接适配权重分布 → 最小化层输出误差

AQLM 的优势在于 codebook 自由度更高（learned vs fixed lattice），且 input-aware + intra-block tuning 提供了额外的误差补偿能力。代价是 codebook 学习需要更多 calibration 时间和计算。

## 部署与生态

- **推理引擎**：HuggingFace Transformers（通过 AQLM 集成），有限的 vLLM 支持
- **量化工具**：官方 PyTorch 实现（GitHub: Vahe1994/AQLM）
- **Calibration 数据**：RedPajama subset，sequence length 4096
- **硬件**：GPU（需专用 CUDA/Triton kernel）；提供 CPU kernel（4× 加速）
- **量化耗时**：显著长于 GPTQ/AWQ——7B 模型需数天（beam search 为主要瓶颈）
- **HuggingFace 集成**：已有预量化模型可直接加载

## 局限与讨论

1. **Calibration 成本极高**：beam search + block fine-tuning 使量化耗时达数天，远超 GPTQ（小时）和 AWQ（分钟）。这限制了其在需要快速量化部署场景中的实用性。
2. **4-bit 优势缩小**：在 4-bit 区间，AQLM 对比 GPTQ/AWQ 的优势不再显著（编码空间充裕时，直接量化已足够好）。AQLM 的核心价值在 2-3 bit 的极限压缩区间。
3. **推理 kernel 支持有限**：虽然理论上 LUT-based 解码可以很快，但生态支持远不如 GPTQ/AWQ（缺乏 vLLM/TensorRT-LLM 的深度集成）。
4. **同质格式优势**：与 SpQR（sparse + quantized hybrid）不同，AQLM 使用简单的同质格式（所有 group 相同编码方式），工程实现更友好。

## 后续工作

AQLM 开创了 LLM 向量量化的路线，后续出现了多项跟进：

- **PV-Tuning**（AQLM v2）：改进 fine-tuning 算法，进一步提升 2-3 bit 精度
- **Efficient Codebook Optimisation**（2025）：加速 codebook 优化过程
- **Ultimate Compression via Meta Networks**（AAAI 2025）：使用 meta network 替代固定 codebook

## 关联页面

- [[Quantization Deployment Formats]]——AQLM 作为极限压缩推荐方案
- [[Weight-Only Quantization]]——AQLM 属于 weight-only 量化但采用 VQ 范式
- [[Lattice Codebooks]]——QuIP# 的 E₈ 固定格码本 vs AQLM 的 learned codebook
- [[Rotation-based Quantization]]——QuIP# 需要旋转预处理，AQLM 不需要
- [[Post-Training Quantization]]——AQLM 超越两步分解框架的范式
