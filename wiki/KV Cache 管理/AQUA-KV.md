---
title: "Cache Me If You Must: Adaptive Key-Value Quantization for Large Language Models"
aliases: [AQUA-KV, Adaptive Quantization, Inter-layer KV Prediction]
source_type: paper
source_url: "https://arxiv.org/abs/2505.xxxxx"
source_date: 2025-05-01
ingested: 2026-05-13
venue: Preprint (under review)
authors: [Alina Shutova, Vladimir Malinovskii, Vage Egiazarian, Denis Kuznedelev, Denis Mazur, Nikita Surkov, Ivan Ermakov, Dan Alistarh]
affiliation: HSE University / Yandex / ISTA / SberDevices / MIPT / T-Bank
code: null
tags: [model-compression, kv-cache, quantization, inter-layer-prediction, residual-quantization, linear-predictor, calibration, compatible-with-pruning]
---

## 核心要点

AQUA-KV 是一种**基于层间依赖的 KV cache 自适应量化**方法。核心洞察：由于 transformer 的残差连接，相邻层的 Key/Value 之间存在强线性可预测性。AQUA-KV 训练紧凑的线性预测器来利用这种依赖关系，然后只**量化残差（residual）**而非原始 KV 值。

![[images/AQUA-KV_fig1_longbench.png]]

关键创新：

1. **层间依赖分析**：系统性地发现前一层 Keys 是当前层 Keys 的最佳预测器（EVR ~0.82），前一层 Values + 当前层 Keys 是 Values 的最佳预测器（EVR ~0.72）
2. **残差量化架构**：Predictor + Quantized Residual = Reconstructed KV，只存储量化后的残差
3. **通用兼容性**：可与 HIGGS、Quanto、KIVI、KVQuant 等任意后端量化器组合
4. **可与 pruning 组合**：与 H2O token eviction 正交兼容，实现 38.3× 内存压缩

结果：在 2-bit 量化下，Llama3.2 3B LongBench 44.26（vs HIGGS 43.25、KIVI 39.63），Llama3.1 70B LongBench 52.79（vs HIGGS 52.18），Qwen2.5 7B LongBench 46.43（vs HIGGS 25.97）——在 BF16 baseline 仅有 2.09 bit/entry 的情况下接近无损。

## 动机：层间线性依赖

### 为什么 KV 可以跨层预测？

Transformer 的残差连接使得相邻层的 hidden states 仅相差一个 transformer block 的增量：

```
h_L = h_{L-1} + TransformerBlock_L(h_{L-1})
```

因此 Key/Value 作为 hidden states 的线性投影，自然继承了这种层间相似性。在 GQA（Grouped-Query Attention）模型中，KV head 数量远少于 Q head，使得预测器参数量极小（FLOPs 不到基础模型的 1/500）。

### 实验验证：Explained Variance Ratio

![[images/AQUA-KV_fig2_variance_ratios.png]]

通过线性 probe 在 Llama-3.2 3B 上测量 Mean Explained Variance Ratio（EVR），发现：

- **Predict Keys**（上图）：前一层 Keys (L-1) EVR ~0.82，远优于 L-2 (~0.74) 和 L-3 (~0.68)；时间维度（past tokens）几乎无帮助（T-1 仅 ~0.42）；混合信号 K_{L-1} 单独即达 ~0.85
- **Predict Values**（下图）：前一层 Values (L-1) EVR ~0.64，加入当前层 Keys 后 KV_{L-1} ~0.72；时间维度同样无帮助（T-1 ~0.17）
- **关键结论**：层间空间依赖 >> 时间依赖；Keys 比 Values 更易预测

残差（prediction error）方差降低为原始方差的 1-EVR 倍。对 scale-independent 量化方法，若 EVR=0.82，则残差量化误差约为直接量化的 ~1/10。

## 方法设计：AQUA-KV Algorithm

### 整体架构

![[images/AQUA-KV_fig3_inference_scheme.png]]

推理时每个 Block L 的流程：

1. **Keys Predictor**：以 Keys Cache_{L-1}（pre-RoPE）为输入，线性投影得到 Keys 预测值
2. **Values Predictor**：以 Values Cache_{L-1} + 当前 Keys 为输入，线性投影得到 Values 预测值
3. **加上残差**：Predicted + Dequantize(Quantized Residual) = Reconstructed KV
4. **存储**：每层只缓存量化后的残差（而非完整 KV）

### Predictor 设计

| 类型 | 参数量 | Wiki2 PPL | LongBench |
|------|--------|-----------|-----------|
| Linear (默认) | 162 MiB | 7.03 | 44.26 |
| MLP (2层) | 540 MiB | 7.03 | 44.61 |
| RRR (低秩) | 68 MiB | 7.03 | 44.61 |
| GPTQ (4-bit) | 41 MiB | 7.16 | 44.19 |

所有架构在精度上几乎等价，默认选择 Linear（最简洁，无额外非线性）。GPTQ 量化 predictor 权重后只需 41MiB，精度几乎无损。

### Pre-RoPE vs Post-RoPE

线性预测器在 **pre-RoPE** 空间效果更好。原因：post-RoPE 下最优预测器需要 rotation-equivariant 结构，而简单线性映射无法满足这一约束。

### 第一层处理

第一层（L=0）没有前一层可参考，采用直接 4-bit 量化（因为实验表明 3-bit 即 near-lossless，2-bit 会有明显降低）。

### Attention Sinks

前 4 个 token（attention sinks）保持不压缩，与文献中的通用做法一致。

## 训练与校准（Algorithm 1）

### 校准流程

1. 从 RedPajama 采样 256 条序列（8192 tokens），其中 32 条用于超参选择，224 条用于训练
2. 对每一层，用 model forward pass 收集前一层和当前层的 KV cache
3. 训练线性回归 predictor（closed-form 解 / simple linear regression）
4. 选择每层量化比特数和 group size（超参搜索）

### 计算开销

- Llama3.2 3B：单 GPU 约 1 小时
- Llama3.1 70B：单 GPU 约 6 小时
- One-shot calibration，无需迭代优化

### 量化器无关性

AQUA-KV 不绑定特定量化后端。校准完成后，可直接搭配任意量化方案（HIGGS、Quanto、KIVI、KVQuant 等）对残差进行量化。

## 实验结果

### Table 1：消融实验（Llama3.2 3B，2-bit）

![[images/AQUA-KV_table1_ablation.png]]

关键消融结论：

- **vs 无预测**：HIGGS 2-bit 直接量化 PPL=9.43 / LongBench=43.25 → AQUA-KV (HIGGS) PPL=7.03 / LongBench=44.26
- **Predictor 架构无关**：Linear/MLP/RRR/GPTQ 精度几乎相同
- **第一层 4-bit 必要**：keep in BF16 或 HIGGS 3/4-bit 差异很小，但 2-bit 明显降低
- **Keys predictor 更重要**：移除 V predictor（PPL 7.06）影响较小，移除 K predictor（PPL 7.43）影响更大
- **Quantizer-agnostic training**：用 agnostic 方式训练的 predictor（PPL 7.13）仅略差于 quantizer-specific training

### Table 2：主要结果（5 模型 × 2/3/4-bit）

![[images/AQUA-KV_table2_main_results.png]]

跨 5 个模型的一致结论：

| 模型 | Bits | AQUA-KV PPL | Best Baseline PPL | AQUA-KV LongBench | Best Baseline LongBench |
|------|------|-------------|-------------------|-------------------|------------------------|
| Llama3.2 3B | 2 | 7.03 | 9.43 (HIGGS) | 44.26 | 43.25 (HIGGS) |
| Llama3.1 8B | 2 | 6.14 | 7.55 (HIGGS) | 44.48 | 44.13 (HIGGS) |
| Llama3.1 70B | 2 | 2.54 | 2.77 (HIGGS) | 52.79 | 52.18 (HIGGS) |
| Qwen2.5 7B | 3 | 6.98 | 6.96 (HIGGS) | 48.30 | 47.57 (HIGGS) |
| Qwen2.5 7B | 2 | 7.55 | 9.63 (HIGGS) | 46.43 | 25.97 (HIGGS) |

特别值得注意的是 Qwen2.5 7B 在 2-bit 下，HIGGS 直接崩溃（LongBench 25.97），而 AQUA-KV 依然保持 46.43（接近 BF16 的 48.13）。

### Table 3：与 H₂O Token Pruning 组合

![[images/AQUA-KV_table3_h2o.png]]

AQUA-KV 与 token eviction 方法正交兼容：

| 方法 | Quant Bits | Memory Saved | LongBench 3B | LongBench 8B |
|------|-----------|-------------|-------------|-------------|
| H₂O only (20% tokens) | 16 | 5.0× | 38.82 | 41.42 |
| H₂O + HIGGS | 2.02 | 39.6× | 37.02 | 40.53 |
| H₂O + AQUA-KV | 2.09 | 38.3× | 38.43 | 40.88 |

组合后实现 38.3× 内存节省，同时 LongBench 分数显著优于 H₂O + HIGGS。

## 推理性能

AQUA-KV 在 backbone quantizer 之上增加额外的 prediction step，因此不可能比 baseline 量化更快。实测：

- Llama3.1 8B：AQUA-KV 在 8K context / batch=1 下约 1300 tok/s（BFloat16 ~1600 tok/s），开销约 18%
- Llama3.1 70B：在更大 batch 下 memory saving 带来的 throughput 增益可抵消 predictor 开销
- 预测器 FLOPs 不到基础模型的 1/500

## 与相关方法的关系

- **[[KV Cache Quantization]]**：AQUA-KV 不是一种独立量化方法，而是在任意量化器之前添加"预测→量化残差"的 wrapper，提升所有量化器的有效精度
- **KIVI / KVQuant**：per-channel / per-token 量化方法，AQUA-KV 可直接搭配使用
- **HIGGS / Quanto**：当前 AQUA-KV 实验中的主要 backbone quantizer
- **[[Token Eviction]]**：与 H2O 等 eviction 方法正交组合（Table 3），实现更极端的压缩比
- **[[DiffKV]]**：同样利用 Key>Value 的差异化观察，但 DiffKV 通过不同精度处理 K/V，而 AQUA-KV 通过层间预测减小需量化的残差
- **[[EVICPRESS]]**：EVICPRESS 联合优化 compression+eviction 的 per-context 策略，AQUA-KV 可作为其 compression 组件之一

## 关键贡献与局限

**贡献：**
- 首次系统性分析 KV cache 的层间线性依赖结构
- 提出通用的 predictor + residual quantization 框架
- 证明简单线性预测器即可匹配复杂非线性模型
- One-shot calibration，无需 per-quantizer 重训
- 与 pruning 正交兼容，极端压缩比下仍保持质量

**局限：**
- 额外 prediction step 增加推理延迟（~18% overhead for 8B model）
- 需要校准数据（256 sequences × 8192 tokens）
- 第一层无法预测，必须保持较高精度（4-bit）
- Predictor 权重需常驻显存（162MiB for linear, 可用 GPTQ 压缩到 41MiB）
