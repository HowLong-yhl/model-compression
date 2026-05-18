---
title: bitsandbytes
aliases: [bnb, bitsandbytes library]
tags: [model-compression, entity, tool]
created: 2026-04-20
updated: 2026-04-20
sources: [LLM.int8()]
---

## 概述

bitsandbytes 是由 Tim Dettmers 开发的开源 Python 库，提供 8-bit 优化器和 8-bit 矩阵乘法原语。它是 [[LLM.int8()]] 方法的参考实现，也是 HuggingFace Transformers 生态中最广泛使用的量化后端之一。

## 核心功能

- **LLM.int8() 推理**：通过 `Linear8bitLt` 模块实现 vector-wise 量化 + mixed-precision decomposition
- **8-bit 优化器**：减少训练时的优化器状态内存占用
- **4-bit NF4 量化**：Normal Float 4-bit，量化点按正态分布最优间隔；配合 double quantization（scale 也量化到 8-bit）进一步压缩。是 **QLoRA** 微调的核心基础设施
- **部署定位**：零配置即时量化，无需 calibration 或预处理步骤。适合快速实验和微调，但推理速度低于 GPTQ/AWQ 的优化 kernel。详见 [[Quantization Deployment Formats]]

## HuggingFace 集成

在 HuggingFace Transformers 中，只需一行代码即可启用：

```python
model = AutoModelForCausalLM.from_pretrained("model_name", load_in_8bit=True)
```

会自动将 feed-forward 和 attention projection 层替换为 `Linear8bitLt` 模块，实现 FP16 → INT8 的无损转换。

## 与本 Wiki 的关联

bitsandbytes 使得 [[LLM.int8()]] 从论文走向实际应用，让 OPT-175B/BLOOM-176B 在消费级 GPU（如 8×RTX 3090）上的推理成为可能。它也是许多后续量化工作（如 QLoRA）的基础设施。

## 链接

- GitHub: https://github.com/TimDettmers/bitsandbytes
- PyPI: `pip install bitsandbytes`
