---
title: Weight-Only Quantization
aliases: [权重量化, Weight Quantization]
tags: [model-compression, quantization, concept]
created: 2026-04-20
updated: 2026-04-20
sources: [GPTQ, AWQ]
---

## 定义

Weight-Only Quantization 是指仅对模型权重进行量化（降低精度），激活保持 FP16/BF16 的量化方式。典型格式为 **W4A16**（4-bit 权重，16-bit 激活）或 **W3A16**。

## 核心优势

与 Weight-Activation 量化（如 [[SmoothQuant]] 的 W8A8）相比：

- **可以压到更低 bit**：权重分布通常比激活更均匀，容忍更激进的量化（3-bit 甚至 2-bit）
- **无需处理激活 outlier**：因为激活保持 FP16，[[Emergent Outlier Features]] 不直接影响推理精度
- **内存节省更大**：W4A16 比 W8A8 额外节省约 2× 模型权重内存

## 核心劣势

- **不加速计算本身**：只减少了权重的内存占用和读取带宽，不减少激活计算的算力需求
- 推理仍是 FP16 矩阵乘法（需要 dequantize 后与 FP16 激活相乘）
- 加速来源于**减少内存传输**——在 memory-bound 的生成式推理中效果显著，但在 compute-bound 场景（如大 batch prefill）中收益有限

## 代表方法

### [[GPTQ]]（ICLR 2023）

基于近似二阶信息（Hessian 逆矩阵）的 one-shot 量化。对每一层的权重，通过 OBS 框架找到使量化误差最小的权重调整。175B 模型在 4 bit 下仅 +0.03 perplexity，在 3 bit 下仍可用。量化需约 4 GPU 小时。

### [[AWQ]]（MLSys 2024 Best Paper）

基于激活感知的等价缩放。识别 0.1%-1% 的 salient weight channels，通过数学等价的 per-channel 缩放保护它们，无需 backpropagation。对多模态模型和领域迁移更鲁棒。仅需 16 个 calibration 样本。

### RTN（Round-to-Nearest）

最简单基线：直接将每个权重四舍五入到最近的量化点。在 4-bit 下尚可接受，3-bit 下完全崩溃。

## GPTQ vs AWQ

| 维度 | [[GPTQ]] | [[AWQ]] |
|------|----------|---------|
| 核心技术 | Hessian-based 二阶重建 | 激活感知等价缩放 |
| Backpropagation | 不需要（仅前向传播计算 Hessian） | **不需要** |
| Calibration 数据量 | 128 segments | **16 sequences** |
| Calibration 鲁棒性 | 较弱（可能过拟合） | **强** |
| 多模态支持 | 退化明显（-6.72 CIDEr） | **近无损**（-1.17 CIDEr） |
| INT4 精度 | 好 | **更好** |
| INT2 能力 | 可用 | 需与 GPTQ 组合 |
| 可组合性 | — | 与 GPTQ 正交可组合 |

两者在实践中经常被组合使用，尤其在极低比特场景下。

## 在部署中的位置

Weight-Only Quantization 主要服务于**边缘/端侧部署**和 **memory-bound 推理**场景：

- 端侧设备（手机、嵌入式 GPU）上内存有限，W4 使 7B 模型可运行
- 单用户低 batch 生成式推理是 memory-bandwidth-bound 的，W4 可实现 3-4× 加速
- 大 batch 服务场景下收益递减，此时 [[SmoothQuant]] 的 W8A8 可能更合适（同时加速计算）
- 极低比特（2-3 bit）场景：[[QuIP#]] 通过 [[Lattice Codebooks]]（E₈ 格码本向量量化）首次证明 3-bit 的 scaling 可优于 4-bit
- [[Quantization Granularity]] 对 weight-only 方法的精度影响显著（group-128 为常用配置）

各格式在工程部署中的详细对比（GPTQ vs AWQ vs GGUF vs EXL2 等）及选型指南见 [[Quantization Deployment Formats]]。各主流模型系列的量化版本现状见 [[Model Quantization Ecosystem]]。
