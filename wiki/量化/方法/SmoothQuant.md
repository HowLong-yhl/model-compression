---
title: "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"
aliases: [SmoothQuant, SQ, Xiao 2022]
source_type: paper
source_url: "https://arxiv.org/abs/2211.10438"
source_date: 2022-11-18
ingested: 2026-04-20
venue: ICML 2023
authors: [Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han]
affiliation: MIT Han Lab
code: "https://github.com/mit-han-lab/smoothquant"
tags: [model-compression, quantization, method, post-training, W8A8, deployment]
---

## 核心要点

SmoothQuant 是一种无需训练的通用 [[Post-Training Quantization]] 方案，通过数学等价变换将量化难度从激活"迁移"到权重，首次实现了对 LLM 的准确高效 **W8A8** 量化。核心洞察：权重容易量化，激活（因 [[Emergent Outlier Features]]）难以量化——通过 per-channel 缩放可以平衡两者的量化难度。

## 方法摘要

### 核心问题

- **权重**分布均匀平坦，INT8 量化几乎无损
- **激活**存在 outlier channels，幅度比正常值大 **~100×**（因模型和层而异，参见 [[Emergent Outlier Features]]），使 per-tensor 量化后非 outlier 通道仅剩 2-3 个有效量化级
- Outlier 是 **channel-wise** 的：固定通道在所有 token 中持续产生 outlier（inter-token 方差小）

### Smoothing 变换

对线性层 `Y = XW`，引入 per-channel 缩放向量 `s ∈ R^{C_i}`：

```
Y = (X · diag(s)⁻¹) · (diag(s) · W) = X̂ · Ŵ
```

- `X̂ = X · diag(s)⁻¹`：将激活的 outlier 通道除以大的 s_j，平滑掉 outlier
- `Ŵ = diag(s) · W`：权重对应通道乘以 s_j，吸收迁移过来的量化难度

输出 Y 数学上**完全相同**，变换本身不引入任何近似。

### Migration Strength α

缩放因子的计算：

```
s_j = max(|X_j|)^α / max(|W_j|)^{1-α}
```

- `α = 1`：所有难度推给权重（激活完美平坦，但权重出现 outlier）
- `α = 0`：所有难度留在激活
- `α = 0.5`：平衡分配，对应通道的激活和权重拥有相似的最大幅度

**实际最优 α 因模型而异：**

| 模型 | α |
|------|---|
| OPT, BLOOM | 0.5 |
| GLM-130B (~30% outlier channels) | 0.75 |
| LLaMA-1 | 0.8 |
| Llama-2-7B/13B | 0.85 |
| Llama-2-70B | 0.9 |
| Falcon-7B | 0.6 |
| Mistral-7B, Mixtral-8x7B | 0.8 |

### 离线融合

`diag(s)⁻¹` 可以**融合到前一层的 LayerNorm 参数中**（离线完成），运行时零开销。当输入来自 residual addition 时需要额外加一个 scaling 层。

### 三个效率级别

| 级别 | 权重量化 | 激活量化 | 备注 |
|------|---------|---------|------|
| **O1** | per-tensor | per-token dynamic | 最准确 |
| **O2** | per-tensor | per-tensor dynamic | 均衡 |
| **O3** | per-tensor | per-tensor static | 最高效 |

## 实验结果

### OPT-175B (7 个 zero-shot 任务平均)

| 方法 | 平均准确率 | WikiText PPL |
|------|-----------|-------------|
| FP16 | 66.9% | 10.99 |
| Naive W8A8 | 35.5% | 93,080 |
| ZeroQuant | 35.8% | 84,648 |
| [[LLM.int8()]] | 66.7% | 11.10 |
| **SmoothQuant-O1** | **66.5%** | 11.11 |
| **SmoothQuant-O3** | **66.8%** | 11.17 |

Naive W8A8 在 OPT-175B 上完全崩溃（准确率近随机），SmoothQuant 在所有级别下均保持 FP16 精度。

### 530B MT-NLG

4 个任务平均准确率：FP16 = 73.1%，INT8 = **73.1%**（完全一致）。SmoothQuant 使 530B 模型在**单节点 8×A100**上服务成为可能（原需 16 GPU / 双节点）。

### 加速与内存

| 实现 | 最大加速 | 内存节省 |
|------|---------|---------|
| PyTorch (单 A100) | 1.51× | 1.96× |
| FasterTransformer | **1.56×** | **~2×** |

对比 [[LLM.int8()]]：后者因混合精度分解开销，延迟几乎总是**慢于 FP16**；SmoothQuant 在所有配置下**都快于 FP16**。

### 多模型验证

在 OPT, BLOOM, GLM-130B, LLaMA-1/2, Falcon, Mistral, Mixtral, MT-NLG 上均验证了近无损的 W8A8 量化。

## Calibration

- **数据集**：The Pile validation set
- **样本数**：512 random sentences
- **用途**：收集 per-channel 激活幅度统计 `max(|X_j|)` 以及 O3 模式的静态量化步长
- α 通过在 Pile 子集上的快速 grid search 选定

## 局限性

- 仅针对 **W8A8 INT8**，不直接解决更低精度（W4A4 等）的挑战
- 需要 per-model 的 α 超参数调优（虽然搜索空间小）
- O3 静态量化依赖 calibration 统计，可能不完美匹配实际推理分布
- Smoothing 只能在有可参数化前驱操作（如 LayerNorm）时无开销融合
- 不同架构量化难度差异大（BLOOM 容易，GLM-130B 困难）

## 与已有知识的关联

- 与 [[LLM.int8()]] 的关系：两者都解决 outlier 导致的激活量化难题，但策略不同——LLM.int8() 在运行时做混合精度分解（准确但有延迟开销），SmoothQuant 在离线做等价变换（无运行时开销）。精度上两者相当，**硬件效率上 SmoothQuant 显著更优**
- 与 [[AWQ]] 共享"[[Equivalent Scaling Transformation|等价缩放变换]]"的核心思路，但方向不同：SmoothQuant 缩放激活通道以平滑 outlier（服务于 W8A8），AWQ 缩放权重通道以保护 salient weights（服务于 W4A16）
- 在两步分解框架中属于 [[Unoptimized Scaling]] + RTN 的组合（参见 [[Post-Training Quantization]]）
- 与 [[GPTQ]] 是互补方法：SmoothQuant 做 W8A8（权重+激活），GPTQ 做 weight-only 到更低 bit
- 已被 NVIDIA TensorRT-LLM、Microsoft ONNX Runtime、Amazon SageMaker、Intel Neural Compressor 等工业框架采纳
- SmoothQuant 的 outlier 迁移思想已被扩展到扩散模型量化：MPQ-DM（AAAI 2025）直接借鉴了其通道级 smooth 策略（详见 [[Diffusion Model Quantization]]）

## 引用信息

原始来源：`raw/SmoothQuant.md`（待录入）
