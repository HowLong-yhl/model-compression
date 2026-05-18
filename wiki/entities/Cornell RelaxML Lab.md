---
title: Cornell RelaxML Lab
aliases: [RelaxML, Cornell CS]
tags: [model-compression, entity, lab, academia]
created: 2026-04-20
updated: 2026-04-20
sources: [QuIP#]
---

## 简介

Cornell RelaxML Lab 是康奈尔大学计算机科学系的研究组，由 Christopher De Sa 领导，专注于高效机器学习系统，尤其在**低精度计算、量化理论、随机化算法**方面有深厚积累。

## 代表工作

| 论文 | 会议 | 贡献 |
|------|------|------|
| QuIP | NeurIPS 2023 | 首次提出 incoherence processing + LDLQ 自适应舍入 |
| **[[QuIP#]]** | **ICML 2024** | RHT + E₈ lattice codebook，2-bit SOTA |

## 核心研究方向

- **Incoherence theory**：从数学上证明了 incoherence 是量化误差的决定性因素
- **Adaptive rounding**：LDLQ 算法族（最优线性反馈自适应舍入）
- **Lattice-based VQ**：将数论中的格结构引入神经网络量化
- **PTQ scaling laws**：首次证明 3-bit 可以比理论无损 4-bit scale 更好

## 关联

- 与 Meta（[[SpinQuant]] 团队）存在密切引用关系：SpinQuant 以 QuIP# 的 Hadamard bound 作为理论依据
- QuIP# 的 incoherence processing 启发了后续旋转量化工作的理论分析
