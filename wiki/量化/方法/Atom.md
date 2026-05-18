---
title: "Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving"
aliases: [Atom, Atom Quantization, Zhao 2023]
source_type: paper
source_url: "https://arxiv.org/abs/2310.19102"
source_date: 2023-10-29
ingested: 2026-05-08
venue: MLSys 2024
authors: [Yilong Zhao, Chien-Yu Lin, Kan Zhu, Haotian Ye, Lequn Chen, Binhang Yuan, Maarten Bosma, Arvind Krishnamurthy, Tianqi Chen]
affiliation: University of Washington, UC Berkeley, Google
code: "https://github.com/efeslab/Atom"
tags: [model-compression, quantization, method, post-training, W4A4, serving, deployment]
---

## 核心要点

Atom 是一种面向 LLM **serving** 场景的低比特权重-激活联合量化方法（W4A4），通过混合精度 outlier 处理、通道重排序、细粒度组量化和 KV cache 量化等技术组合，在保持近无损精度的同时大幅提升推理吞吐量。核心洞察：serving 场景（大 batch）中，仅做 weight-only 量化（如 GPTQ/AWQ 的 W4A16）无法充分加速计算密集型的 GEMM——必须**同时量化权重和激活到 INT4** 才能真正利用低比特 Tensor Core 的算力优势。

## 方法摘要

### 核心问题

LLM serving 的推理瓶颈取决于 batch size：

- **小 batch**（单用户）：memory-bound，weight-only 量化（W4A16）有效
- **大 batch**（高并发）：compute-bound，需要 W4A4 利用 INT4 Tensor Core 加速 GEMM

但激活的 INT4 量化极具挑战——激活中存在的 [[Emergent Outlier Features|outlier channels]] 使得均匀 INT4 量化会导致严重的精度崩溃。Atom 通过一系列精度保护技术解决这一问题。

### 技术一：混合精度 Outlier 处理 + 通道重排序（Reorder）

Atom 识别出激活中的 outlier channels（幅度远大于其他通道），将其保留在更高精度（FP16），其余通道量化到 INT4。

**关键创新——通道重排序（Reorder）**：

```
原始布局：[normal, outlier, normal, normal, outlier, normal, ...]
                    ↓ Reorder
重排后布局：[normal, normal, normal, normal, ..., outlier, outlier]
            └────── INT4 量化区域 ──────┘   └── FP16 区域 ──┘
```

- 将 outlier channels **集中到矩阵末尾**
- 对应的权重列同步重排
- GPU kernel 可以对前面的大块连续区域执行 INT4 GEMM，对末尾小块执行 FP16 GEMM
- 重排后两部分的结果相加得到最终输出

这比 [[LLM.int8()]] 的运行时动态分解更高效——重排是 offline 的静态操作，推理时无需逐 token 判断 outlier。

### 技术二：细粒度组量化（Fine-grained Group Quantization）

对激活使用 **per-group asymmetric INT4 量化**：

- Group size：128（与权重 group size 对齐）
- 非对称量化（asymmetric）：计算 per-group scale 和 zero-point
- **动态量化**（dynamic quantization）：每次推理时根据当前 batch 的实际激活值在线计算量化参数

相比 per-tensor 量化，per-group 显著降低了组内数值范围差异带来的精度损失。

### 技术三：权重量化

权重使用 **per-group INT4** 量化，配合 GPTQ 式的 Hessian 误差补偿：

- Group size：128
- 使用 [[GPTQ]] 的逐列更新策略补偿量化误差
- 权重量化是 offline 完成的（PTQ），推理时无额外开销

在两步分解框架中，Atom 的 Step 1 为 **Reorder**（通道重排序），Step 2 为 **GPTQ**（Hessian 补偿）。

### 技术四：KV Cache 量化

将 KV cache 量化到 INT4，释放显存以支持更大的 batch size：

- Key cache：per-channel INT4（outlier 沿通道维度稳定）
- Value cache：per-token INT4
- 结合 group quantization 进一步细化

KV cache 的压缩使得在相同 GPU 显存下可以服务 **更多并发请求**，直接提升 serving 吞吐量。

### 技术五：定制 INT4 GEMM Kernel

基于 CUTLASS 构建高效的 INT4 混合精度 GEMM kernel：

- 支持 INT4 weight × INT4 activation 的 Tensor Core 计算
- 处理 per-group scale/zero-point 的去量化逻辑
- 融合 outlier 通道的 FP16 分支
- 针对不同 batch size 和矩阵形状做 kernel selection

## 实验结果

### 精度保持

| 模型 | FP16 PPL | Atom W4A4 PPL | Δ |
|------|----------|---------------|---|
| LLaMA-7B | 5.68 | ~6.12 | +0.44 |
| LLaMA-13B | 5.09 | ~5.45 | +0.36 |
| LLaMA-2-7B | 5.47 | ~6.12 | +0.65 |
| LLaMA-2-70B | 3.32 | ~3.68 | +0.36 |

W4A4 下 PPL 退化控制在可接受范围内，远优于 naive INT4 量化（PPL 爆炸）。

### 吞吐量提升

在 A100 GPU 上，serving 场景：

| 对比基线 | Atom 吞吐量提升 |
|----------|----------------|
| vs FP16 | **最高 7.7×** |
| vs SmoothQuant W8A8 | **~2×** |
| vs Weight-only W4A16 (大 batch) | **~1.8×** |

- 大 batch（高并发）下优势最显著——此时 GEMM 是 compute-bound，INT4 Tensor Core 直接翻倍算力
- 小 batch 下优势减小——memory-bound 场景中 W4A16 已足够

### 端到端 Serving 指标

结合 KV cache 量化带来的显存节省（可容纳更大 batch），Atom 在端到端 serving 吞吐量（tokens/second under SLA）上实现了 **3.1-7.7×** 的提升（vs FP16 baseline）。

## Calibration

- **数据集**：WikiText-2 / C4
- **样本数**：128 sequences
- **用途**：GPTQ 权重量化的 Hessian 计算；识别 outlier channels 的位置

## 局限性

- 需要 **INT4 Tensor Core**（仅 NVIDIA Ada/Hopper 架构，即 RTX 4090/H100 等）——硬件要求高于 [[QServe]]（仅需 INT8 TC）
- 通道重排序引入额外的 index mapping 开销（虽然可离线完成）
- W4A4 的精度损失仍高于 W4A8 方案（如 [[QServe]] 的 QoQ）——在精度敏感场景下可能不够
- 动态量化的在线 scale 计算有额外开销（per-group reduction）
- 对 outlier channel 数量的假设是固定的——如果模型的 outlier 分布异常，需要调整
- MoE 架构支持有限

## 与已有知识的关联

- **与 [[QServe]] 的关系**：QServe（MLSys 2025）是 Atom 的直接后续竞争者。QServe 选择 W4A8KV4（而非 W4A4）避免了激活 INT4 的精度损失，并通过渐进式量化利用更普及的 INT8 Tensor Core。QServe 在精度上优于 Atom（PPL 5.62 vs 6.12），且硬件门槛更低（A100 即可 vs 需要 Ada/Hopper 的 INT4 TC）
- **与 [[LLM.int8()]] 的关系**：两者都采用 outlier 通道的混合精度处理。LLM.int8() 在运行时动态分解（准确但有延迟），Atom 通过离线重排序实现静态分离（更高效但不够灵活）
- **与 [[SmoothQuant]] 的关系**：SmoothQuant 通过等价缩放消除 outlier 实现 W8A8；Atom 通过重排序+混合精度处理 outlier 实现 W4A4。两者思路互补——SmoothQuant "消灭" outlier，Atom "隔离" outlier
- **与 [[QuaRot]] 的关系**：QuaRot 通过 Hadamard 旋转消除 outlier 实现 W4A4KV4，无需混合精度但需要在线 Hadamard 变换（~8% 开销）；Atom 保留 outlier 但隔离处理。QuaRot 在精度上优于 Atom（PPL ~5.97 vs 6.12）
- **在两步分解框架中的位置**：Step 1 = Reorder（通道重排序，本质是一种无损的预处理变换），Step 2 = GPTQ（Hessian 补偿）。见 [[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 的分类表
- **与 [[Emergent Outlier Features]] 的关系**：Atom 的设计前提是 outlier channels 的存在——这些 channels 空间位置固定（跨 token 稳定），使得离线重排序成为可能
- **与 [[KV Cache Quantization]] 的关系**：Atom 的 KV4 量化粒度选择（Key per-channel、Value per-token）与 [[KIVI]] 的独立发现一致

## 历史定位

Atom 是首批将 **W4A4 + 系统优化** 推向实际可部署 serving 系统的工作之一（MLSys 2024），证明了在大 batch serving 场景下，weight-activation 联合量化比 weight-only 量化能带来更大的吞吐提升。它开创的"量化算法 + 定制 kernel + serving 系统"协同设计范式被后续的 [[QServe]]、QuaRot-serving 等工作继承和改进。

## 引用信息

```bibtex
@inproceedings{zhao2024atom,
  title={Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving},
  author={Zhao, Yilong and Lin, Chien-Yu and Zhu, Kan and Ye, Haotian and Chen, Lequn and Yuan, Binhang and Bosma, Maarten and Krishnamurthy, Arvind and Chen, Tianqi},
  booktitle={Proceedings of Machine Learning and Systems (MLSys)},
  year={2024}
}
```
