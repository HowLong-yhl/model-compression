---
title: "QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving"
aliases: [QServe, QoQ, W4A8KV4]
source_type: paper
source_url: "https://arxiv.org/abs/2405.04532"
source_date: 2024-05-07
ingested: 2026-05-08
venue: MLSys 2025
authors: [Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, Song Han]
affiliation: MIT Han Lab
code: "https://github.com/mit-han-lab/omniserve"
tags: [model-compression, quantization, method, post-training, W4A8, KV-cache, serving, deployment]
---

## 核心要点

QServe 提出了 **QoQ（quattuor-octo-quattuor）** 量化算法——一种 W4A8KV4 方案，以及配套的 GPU 推理系统。核心洞察：LLM serving 的效率瓶颈不仅在于量化位宽，更在于**去量化开销**（dequantization overhead）——naive INT4 weight 虽然压缩了存储，但 INT4→FP16 的去量化计算在 CUDA core 上成为新瓶颈。QoQ 通过**渐进式量化**（progressive group quantization）将 W4 权重先解码到 INT8 而非 FP16，从而利用 INT8 Tensor Core GEMM 获得最佳吞吐。

## 方法摘要

### 核心问题：去量化开销

传统 W4A16 方法（GPTQ/AWQ）的推理路径：

```
INT4 weight → dequant → FP16 → FP16 Tensor Core GEMM
```

在 serving 场景（大 batch、高并发）下，去量化计算的 CUDA core 开销与 Tensor Core GEMM 计算量可比，成为实际吞吐量瓶颈。更极端的 W4A4 方法虽然理论上计算量更小，但 INT4 Tensor Core 仅在最新硬件上可用，且精度损失显著。

QServe 选择 W4A8KV4 作为最优工程折中——权重 INT4 压缩存储、激活 INT8 利用 Tensor Core、KV cache INT4 节省显存。

### QoQ 算法：渐进式组量化（Progressive Group Quantization）

QoQ 对权重的量化分为两级：

```
原始 FP16 权重
    │
    ▼ (Step 1: Per-channel INT8 量化)
  INT8 权重 + FP16 per-channel scale
    │
    ▼ (Step 2: Per-group INT4 量化)
  INT4 权重 + INT8 per-group scale + INT8 per-group zero-point
```

**Step 1**：先对每个输出通道做 per-channel INT8 量化，使用**保护范围 [-119, 119]**（而非满范围 [-127, 127]），预留 headroom 给后续 per-group 细化。

**Step 2**：在 INT8 域内，对每组（group_size 通常 128）进一步量化到 INT4。此时 per-group scale 和 zero-point 本身就是 INT8 值。

**推理时去量化路径**：

```
INT4 → (INT8 per-group dequant) → INT8 → INT8 Tensor Core GEMM
```

关键优势：INT4→INT8 去量化仅涉及整数 shift 和加法（极轻量），避免了 INT4→FP16 的高开销转换。

### SmoothAttention：KV4 的精度保护

将 KV cache 从 FP16 压缩到 INT4 的主要挑战是 Key cache 中存在 outlier 通道（类似权重/激活中的 outlier 问题）。QoQ 引入 SmoothAttention：

```
原始：Attn = softmax(Q · K^T / √d) · V

SmoothAttention：
  K̂ = K · diag(s)⁻¹    ← 缩小 Key outlier 通道
  Q̂ = Q · diag(s)      ← 对应放大 Query 通道
  Attn = softmax(Q̂ · K̂^T / √d) · V    ← 数学等价
```

缩放因子 `s` 通过 calibration 数据上的 Key channel 幅度统计确定，然后**融合到前一层的线性投影权重中**（offline，运行时零开销）——与 [[SmoothQuant]] 的 LayerNorm 融合思想一脉相承。

### 量化粒度

| 组件 | 比特宽度 | 粒度 | 说明 |
|------|---------|------|------|
| Weight | INT4 | per-group (128) | 渐进式两级量化 |
| Activation | INT8 | per-token dynamic | 标准动态量化 |
| Key cache | INT4 | per-channel | outlier 沿通道维度聚集 |
| Value cache | INT4 | per-token | value 分布按 token 变化 |

KV cache 的量化粒度选择（Key per-channel、Value per-token）与 [[KIVI]] 和 [[KVQuant]] 的独立发现一致。

## 系统协同设计

QServe 的系统设计针对 QoQ 的特性深度优化 CUDA kernel：

### Compute-Aware Weight Reordering

重新排列 INT4 权重的内存布局，使 INT4 unpacking（从一个 byte 中提取两个 4-bit 值）与 Tensor Core 的 warp-level 数据访问模式对齐，最大化寄存器级并行度。

### Subtraction After Multiplication

传统 asymmetric 量化在 GEMM 前需要减去 zero-point：`(q - z) × s`。QServe 将 zero-point 校正移到 GEMM epilogue 中（矩阵乘法完成后再减），减少了 GEMM 主循环中的指令数。

### Register-Level Parallelism

INT4 → INT8 的 unpacking 使用寄存器级位操作并行化——一条指令同时处理多个元素的去量化，隐藏延迟。

### Attention Kernel 优化

对 KV4 的 attention 计算使用 FP16 算术 + bit-trick dequantization 的融合 kernel，保持整个 attention kernel memory-bound（而非 compute-bound），充分利用 GPU 带宽。

## 实验结果

### 吞吐量对比（tokens/second，serving 场景）

**在 A100-80GB 上（vs TensorRT-LLM W4A16）：**

| 模型 | TensorRT-LLM | QServe | 加速比 |
|------|-------------|--------|--------|
| Llama-3-8B | ~3000 | ~5200 | **1.7×** |
| Llama-2-70B | ~400 | ~700 | **1.8×** |
| Qwen-1.5-72B | ~350 | ~680 | **1.9×** |

**在 L40S 上（vs TensorRT-LLM W4A16）：**

| 模型 | TensorRT-LLM | QServe | 加速比 |
|------|-------------|--------|--------|
| Llama-3-8B | ~2200 | ~5500 | **2.5×** |
| Yi-34B | ~550 | ~1350 | **2.4×** |
| Llama-2-70B | ~200 | ~470 | **2.4×** |

QServe 在 A100 上实现 **1.2-2.4×** 吞吐提升，在 L40S 上实现 **1.5-3.5×** 吞吐提升（vs TensorRT-LLM FP16/W4A16）。

### 精度保持

| 模型 | FP16 PPL | QoQ W4A8KV4 PPL | Δ |
|------|----------|-----------------|---|
| Llama-2-7B | 5.47 | 5.62 | +0.15 |
| Llama-2-70B | 3.32 | 3.42 | +0.10 |
| Llama-3-8B | 6.14 | 6.32 | +0.18 |
| Mistral-7B | 5.25 | 5.39 | +0.14 |

精度损失极小，明显优于 W4A4 方法（如 [[Atom]]、[[QuaRot]]），后者在同等模型上通常有 0.5-2.0 的 PPL 退化。

### 与 W4A4 方法的对比

| 方法 | 格式 | Llama-2-7B PPL | 硬件要求 |
|------|------|---------------|---------|
| [[Atom]] | W4A4 | 6.12 | INT4 TC |
| QuaRot | W4A4 | 5.97 | INT4 TC |
| **QoQ** | **W4A8KV4** | **5.62** | INT8 TC（更普及） |

QoQ 在精度上优于 W4A4 方法，同时仅需 INT8 Tensor Core（A100/L40S 即可），无需 INT4 Tensor Core（仅 Ada/Hopper 以上支持）。

## Calibration

- **数据集**：WikiText-2
- **样本数**：128 sequences
- **用途**：确定 per-channel INT8 scales、per-group INT4 scales/zero-points、SmoothAttention 的通道缩放因子

## 局限性

- 仅支持 NVIDIA GPU（需 INT8 Tensor Core，即 sm_75+ Turing 及以上）
- 渐进式量化的保护范围 [-119, 119] 是经验选择，理论最优性未证明
- SmoothAttention 的缩放因子为静态（calibration 时确定），对分布漂移可能非最优
- 相比 W4A16（GPTQ/AWQ），增加了激活量化开销；但在 serving 大 batch 场景下收益远大于开销
- 当前实现对 MoE 架构（如 Mixtral）的支持有限

## 与已有知识的关联

- **与 [[SmoothQuant]] 的关系**：SmoothAttention 直接继承了 SmoothQuant 的等价缩放思想——SmoothQuant 平滑 FFN 激活 outlier，SmoothAttention 平滑 Key cache outlier。二者都将缩放因子融合进前一层权重实现零运行时开销
- **与 [[AWQ]] 的关系**：同出 [[MIT Han Lab]]，AWQ 面向 W4A16 weight-only 场景，QServe 面向 W4A8KV4 serving 场景。AWQ 的 TinyChat 优化单用户推理，QServe 优化高并发服务
- **与 [[KV Cache Quantization]] 的关系**：QoQ 的 KV4 量化（Key per-channel、Value per-token）与 [[KIVI]] 的实验发现完全一致，但 QoQ 额外引入 SmoothAttention 提升 KV4 精度
- **与 [[QuaRot]] / [[SpinQuant]] 的关系**：QuaRot/SpinQuant 追求 W4A4KV4 全面低比特化，依赖旋转消除 outlier；QServe 选择 W4A8 保留 INT8 激活精度，换取更好的精度-吞吐平衡
- **与 [[Quantization Deployment Formats]] 的关系**：QServe 是首个将 W4A8KV4 从论文推向可部署 serving 系统的工作，补充了 GPTQ/AWQ 生态中缺失的"量化感知服务端"
- **在两步分解框架中的位置**：QoQ 的权重部分对应 [[Unoptimized Scaling]]（per-channel + per-group）+ RTN；SmoothAttention 对应 KV 域的 [[Equivalent Scaling Transformation]]

## 引用信息

```bibtex
@inproceedings{lin2025qserve,
  title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
  author={Lin, Yujun and Tang, Haotian and Yang, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
  booktitle={Proceedings of Machine Learning and Systems (MLSys)},
  year={2025}
}
```
