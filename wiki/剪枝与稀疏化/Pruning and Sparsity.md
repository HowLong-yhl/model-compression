---
title: Pruning & Sparsity
aliases: [剪枝, 稀疏化, Model Pruning, Weight Pruning]
tags: [model-compression, concept, pruning, sparsity]
created: 2026-04-28
updated: 2026-04-28
sources: [SparseGPT, Wanda]
---

## 核心概念

剪枝（Pruning）将模型权重中不重要的参数置零或移除，减少模型大小和计算量。与量化（改变数值精度）正交——量化保留所有参数但用更少比特存储，剪枝直接减少参数数量。

## 基本分类

### 结构化 vs 非结构化

- **非结构化剪枝**：任意位置的权重可被置零，产生稀疏矩阵。压缩灵活但需要稀疏计算硬件支持（如 NVIDIA Ampere 的 2:4 sparsity）
- **结构化剪枝**：按行/列/head/layer 整体移除，直接减少矩阵维度。硬件友好但压缩粒度粗

### Post-Training vs Training-time

类比量化的 PTQ vs QAT，剪枝也分为训练后一次性剪枝（如 [[SparseGPT]]、[[Wanda]]）和训练过程中逐步剪枝。LLM 时代以 post-training 为主流。

## 代表方法

### SparseGPT（ICML 2023, IST Austria）

[[SparseGPT]] 是首个能在单次处理中将 OPT-175B / BLOOM-176B 剪枝到 50-60% 稀疏度且几乎不损失精度的方法。核心技术与 [[GPTQ]] 共享 Hessian-based 框架——逐列处理权重，将被剪枝权重的误差通过 Hessian 信息补偿到未剪枝权重上。

### Wanda（ICLR 2024, CMU）

[[Wanda]] 提出极简的剪枝准则——**权重绝对值 × 对应激活幅度**。无需 Hessian 计算，比 SparseGPT 快 ~30×，精度相当。核心洞察：与 [[AWQ]] 的激活感知缩放异曲同工——激活幅度大的通道对应的权重更重要（直接利用了 [[Emergent Outlier Features]]）。

## 与量化的关系

剪枝和量化解决不同维度的压缩问题，且可组合：

| 维度 | 量化 | 剪枝 |
|------|------|------|
| 压缩对象 | 数值精度（FP16 → INT4） | 参数数量（移除零值） |
| 存储格式 | 低比特整数/浮点 | 稀疏矩阵（CSR/2:4 等） |
| 硬件需求 | 低比特算子（INT4 GEMM） | 稀疏算子（2:4 sparsity） |
| 理论下界 | Rate-distortion theory | 信息论最优稀疏度 |

SparseGPT 论文展示了 50% 稀疏 + 4-bit 量化的组合，在 OPT-175B 上 PPL 退化极小。

## 开放方向

本 wiki 目前以量化和 KV cache 管理为主要深度覆盖。剪枝领域的后续扩展方向包括：SparseGPT/Wanda 的后续改进（DSnoT、RIA、OWL）、N:M 结构化稀疏（2:4, 4:8）、与量化的联合优化、以及稀疏 attention 方法（与 KV cache eviction 的联系）。

## 引用关系

← 已录入来源：[[SparseGPT]]（Hessian-based 一次性剪枝），[[Wanda]]（权重×激活幅度准则）
← 与量化的共享框架：[[GPTQ]]（SparseGPT 与 GPTQ 共享 Hessian 技术），[[AWQ]]（Wanda 与 AWQ 共享激活感知思想）
← 相关概念：[[Emergent Outlier Features]]（Wanda 直接利用 outlier channel 的激活幅度）
→ 概念关联：[[KV Cache Management]]（正交维度），[[Post-Training Quantization]]（类比关系）
