---
title: 四篇经典 PTQ 方法对比
aliases: [方法对比, Classic PTQ Comparison]
tags: [model-compression, quantization, comparison]
created: 2026-04-20
updated: 2026-04-20
sources: [LLM.int8(), GPTQ, SmoothQuant, AWQ]
---

## 概述

本页对比本 wiki 收录的四篇奠基性 LLM 量化论文。它们发表于 2022-2024 年间，共同定义了 LLM [[Post-Training Quantization]] 的技术版图。

## 基本信息

| | [[LLM.int8()]] | [[GPTQ]] | [[SmoothQuant]] | [[AWQ]] |
|---|---|---|---|---|
| **会议** | NeurIPS 2022 | ICLR 2023 | ICML 2023 | MLSys 2024 (Best Paper) |
| **团队** | UW / Meta AI | IST Austria | [[MIT Han Lab]] | [[MIT Han Lab]] |
| **量化类型** | W8A8 | Weight-only (W4/W3) | W8A8 | Weight-only (W4/W3) |
| **核心技术** | Mixed-precision decomposition | 二阶 Hessian 重建 | 等价变换平滑激活 | 激活感知等价缩放 |
| **需要 backprop** | 否 | 否（仅前向传播计算 Hessian） | 否 | 否 |

## 技术路线对比

### 处理 Outlier 的策略

四篇论文都必须应对 [[Emergent Outlier Features]]，但策略截然不同：

| 方法 | 何时处理 | 如何处理 | 开销 |
|------|---------|---------|------|
| [[LLM.int8()]] | **运行时** | 将 outlier 维度提取到 FP16 分支 | 有延迟开销（小模型显著） |
| [[SmoothQuant]] | **离线** | 等价变换使激活均匀化 | 零运行时开销（融入 LayerNorm） |
| [[GPTQ]] | **离线** | 不直接处理，通过 Hessian 补偿全局误差 | 量化时间较长 |
| [[AWQ]] | **离线** | 利用 outlier 信息识别并保护 salient weights | 量化快，仅需 grid search |

### 量化精度 vs 硬件效率

```
更高精度（W8A8）                              更低精度（W4/W3/W2）
├── LLM.int8()  ──  准确但慢               ├── GPTQ  ──  Hessian 重建，可达 2-bit
├── SmoothQuant ──  准确且快               ├── AWQ   ──  缩放保护，鲁棒性强
│   → 适合大 batch 服务                      │   → 适合边缘/端侧部署
│   → 同时节省内存和计算                      │   → 主要节省内存（memory-bound 加速）
```

## 精度对比

### OPT-175B / BLOOM-176B

| 方法 | 精度格式 | OPT-175B Wiki PPL | BLOOM-176B Wiki PPL |
|------|---------|-------------------|---------------------|
| FP16 基线 | FP16 | 8.34 | 8.11 |
| [[LLM.int8()]] | W8A8 | ~8.34 (零损失) | ~8.11 |
| [[SmoothQuant]]-O3 | W8A8 | 11.17* | — |
| [[GPTQ]] | W4 | **8.37** | **8.21** |
| [[GPTQ]] | W3 | 8.68 | 8.64 |
| [[AWQ]] | W4 | — | — |

*SmoothQuant 在 OPT-175B 上的数据使用其论文自身的 FP16 基线（PPL 10.99），与 GPTQ 的基线（PPL 8.34）来自不同评测设置（context length/stride 差异），两者不可直接比较绝对值。SmoothQuant-O3 相对其自身基线的退化为 +0.18。

### LLaMA-7B (WikiText-2 PPL, W4A16, g128)

| FP16 | RTN | GPTQ | **AWQ** |
|------|-----|------|---------|
| 5.68 | 5.96 | 6.22 | **5.78** |

AWQ 在 INT4 下一致优于 GPTQ。

## 效率对比

### 推理加速

| 方法 | 典型加速比 | 原理 |
|------|-----------|------|
| [[LLM.int8()]] | ~1× 或更慢（小模型） | Mixed-precision 分解有开销 |
| [[SmoothQuant]] | **1.5×** (FasterTransformer) | INT8 GEMM + 减少内存 |
| [[GPTQ]] | **3.2-4.5×** | 大幅减少内存传输 + 减少 GPU 数 |
| [[AWQ]] + TinyChat | **3.5-3.7×** | 4-bit kernel fusion + weight packing |

### 量化速度

| 方法 | 175B 模型量化时间 | Calibration 数据量 |
|------|------------------|-------------------|
| [[LLM.int8()]] | 即时（运行时动态） | 无需离线 calibration |
| [[SmoothQuant]] | 分钟级 | 512 sequences |
| [[GPTQ]] | ~4 GPU hours | 128 × 2048 tokens |
| [[AWQ]] | 分钟级 | 16 sequences |

## 互补关系

这四种方法并非互相替代，而是**互补**的：

```
                    W8A8 精度保持
                    ┌─────────────────┐
                    │  SmoothQuant     │ ← 服务端大 batch 推理
                    │  (快速, 通用)    │
                    └─────────────────┘
                            │
                            │ 需要更高压缩比？
                            ▼
                    W4A16 极致压缩
                    ┌─────────────────┐
                    │  AWQ + GPTQ      │ ← 端侧/边缘部署
                    │  (可组合使用)    │
                    └─────────────────┘
                            │
                    ┌───────┴───────┐
                    │ LLM.int8()    │ ← 零配置快速部署
                    │ (零 calibration│    (load_in_8bit=True)
                    │  即插即用)     │
                    └───────────────┘
```

- **SmoothQuant** 适合追求 throughput 的服务端场景（大 batch、高吞吐）
- **AWQ / GPTQ** 适合内存受限的端侧场景（小 batch、低延迟）
- **LLM.int8()** 适合快速原型验证（无需 calibration，即插即用）
- AWQ 和 GPTQ 可**组合**使用（AWQ 缩放 + GPTQ 重建），在 INT2 极端场景下效果最佳

## 时间线

```
2022-08  LLM.int8()   ── 发现 outlier features，提出混合精度分解
2022-10  GPTQ         ── 首次实现 175B 模型 4-bit one-shot 量化
2022-11  SmoothQuant  ── 等价变换解决 W8A8 量化，实现实际加速
2023-06  AWQ          ── 激活感知缩放，鲁棒性和多模态支持突破
```

LLM.int8() 的 outlier 发现为后续所有工作提供了核心动机。
