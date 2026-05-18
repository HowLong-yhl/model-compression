# Index

本索引按分类列出 wiki 中的所有页面。每次 ingest 或新建页面后更新。

> 最后更新：2026-04-20 | 来源数：10 | 总页面数：28

---

## 总览

| 页面 | 说明 |
|------|------|
| [[wiki/overview]] | Wiki 总览 — 量化领域的全局图景、两步分解框架与知识地图 |

## 概念 (Concepts)

| 页面 | 说明 | 关联来源 |
|------|------|---------|
| [[wiki/concepts/Post-Training Quantization]] | PTQ 范式总览：定义、三大分支、两步分解框架 | 全部 5 篇 |
| [[wiki/concepts/Emergent Outlier Features]] | 大模型涌现异常特征：定义、相变、对量化的影响 | LLM.int8(), SmoothQuant, AWQ |
| [[wiki/concepts/Weight-Only Quantization]] | 权重量化范式：GPTQ vs AWQ 对比、部署场景 | GPTQ, AWQ |
| [[wiki/concepts/Equivalent Scaling Transformation]] | 等价缩放变换：SmoothQuant 与 AWQ 的共享核心技术，与 Rotation 的互补关系 | SmoothQuant, AWQ, 综述 |
| [[wiki/concepts/Rotation-based Quantization]] | 正交旋转量化：Hadamard 变换消除 outlier，W4A4 的关键推动力 | 综述, SpinQuant, QuIP# |
| [[wiki/concepts/FP4 Quantization]] | FP4 浮点量化：MXFP4/NVFP4 格式、E2M1 编码、与 INT4 的对比 | 综述 |
| [[wiki/concepts/Quantization Granularity]] | 量化粒度：per-tensor/channel/group 的性能-开销权衡 | 综述 |
| [[wiki/concepts/Lattice Codebooks]] | 格码本向量量化：E₈ 格最优球堆积、E8P 码本设计、与旋转的协同 | QuIP# |
| [[wiki/concepts/Unoptimized Rotation]] | 未优化旋转：随机 Hadamard/RHT，零数据但有方差 | QuIP#, QuaRot |
| [[wiki/concepts/Optimized Rotation]] | 优化旋转：Cayley SGD/Riemann Adam，Stiefel 流形学习 | SpinQuant, OSTQuant, FlatQuant |
| [[wiki/concepts/Unoptimized Scaling]] | 未优化缩放：公式/Grid search，快速鲁棒 | SmoothQuant, AWQ |
| [[wiki/concepts/Optimized Scaling]] | 优化缩放：可学习参数，梯度优化，与旋转联合最优 | OSTQuant, FlatQuant |
| [[wiki/concepts/Quantization Insights]] | LLM 量化实验 Insights：跨论文的关键实验发现与元原则 | 综述, 全部来源 |

## 实体 (Entities)

| 页面 | 说明 | 关联来源 |
|------|------|---------|
| [[wiki/entities/MIT Han Lab]] | Song Han 实验室，SmoothQuant 和 AWQ 的出处 | SmoothQuant, AWQ |
| [[wiki/entities/bitsandbytes]] | LLM.int8() 的开源实现库，HuggingFace 集成 | LLM.int8() |
| [[wiki/entities/Cornell RelaxML Lab]] | Christopher De Sa 实验室，QuIP/QuIP# 的出处 | QuIP# |

## 来源摘要 (Sources)

| 页面 | 标题 | 会议 | 类型 |
|------|------|------|------|
| [[wiki/sources/LLM.int8()]] | LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale | NeurIPS 2022 | W8A8 方法 |
| [[wiki/sources/GPTQ]] | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | ICLR 2023 | Weight-only 方法 |
| [[wiki/sources/SmoothQuant]] | SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs | ICML 2023 | W8A8 方法 |
| [[wiki/sources/AWQ]] | AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | MLSys 2024 | Weight-only 方法 |
| [[wiki/sources/A Comprehensive Evaluation on Quantization Techniques for LLMs]] | A Comprehensive Evaluation on Quantization Techniques for LLMs | arXiv 2025 | **综述/评测** |
| [[wiki/sources/SpinQuant]] | SpinQuant: LLM Quantization with Learned Rotations | ICLR 2025 | Rotation 方法（学习旋转） |
| [[wiki/sources/QuIP#]] | QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks | ICML 2024 | Rotation + VQ 方法 |
| [[wiki/sources/QuaRot]] | QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs | NeurIPS 2024 | Rotation 方法（随机旋转） |
| [[wiki/sources/OSTQuant]] | OSTQuant: Refining LLM Quantization with Orthogonal and Scaling Transformations | arXiv 2025 | 联合优化 Rotation + Scaling |
| [[wiki/sources/FlatQuant]] | FlatQuant: Flatness Matters for LLM Quantization | ICML 2025 | 联合优化 Rotation + Scaling |

## 对比与分析 (Comparisons)

| 页面 | 说明 |
|------|------|
| [[wiki/comparisons/四篇经典 PTQ 方法对比]] | LLM.int8() / GPTQ / SmoothQuant / AWQ 全方位对比 |
