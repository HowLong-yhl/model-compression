---
title: FP4 Quantization
aliases: [FP4 量化, MXFP4, NVFP4, E2M1]
tags: [model-compression, quantization, concept, hardware, FP4]
created: 2026-04-20
updated: 2026-04-20
sources: [A Comprehensive Evaluation on Quantization Techniques for LLMs]
---

## 定义

FP4 Quantization 是指使用 4-bit 浮点数（而非 4-bit 整数 INT4）作为量化数据类型的方法。每个 FP4 值采用 **E2M1 格式**：1 个符号位 + 2 个指数位 + 1 个尾数位。与 INT4 的均匀量化点不同，FP4 的量化点是非均匀分布的。

## E2M1 编码

```
value = (-1)^S × 2^(E-bias) × (1 + 2^(-m) × M)   (E > 0)
value = (-1)^S × 2^(1-bias) × (0 + 2^(-m) × M)    (E = 0, subnormal)
```

其中 bias = 1，因此 FP4 的可表示值为：

`[-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]`

## FP4 vs INT4

| 维度 | INT4 | FP4 (E2M1) |
|------|------|------------|
| 量化点分布 | **均匀** | **非均匀**（对数间距） |
| 长尾分布适配 | 差（大幅值用更多量化点） | **好**（大幅值间距大，小幅值间距细） |
| 适合的数据分布 | 均匀分布 | 正态/长尾分布 |
| 硬件支持 | 广泛 | NVIDIA Blackwell 架构 |

FP4 更适合 LLM 的权重和激活分布（近似正态），尤其在 **无预量化变换** 的条件下优势明显。

## 两种硬件格式

### MXFP4（MX Microscaling FP4）

- 由 Open Compute Project 定义的开放标准
- Group size = **32**
- Scaling factor 格式：**FP8 (E8M0)**——只有 8 个指数位，无尾数，因此 scale 只能是 2 的幂
- 一级缩放方案

### NVFP4

- NVIDIA 专有格式，用于 Blackwell 架构 Tensor Core
- Group size = **16**（更细粒度）
- Scaling factor 格式：**FP8 (E4M3)**——有分数精度，可表达更精细的缩放
- **两级缩放方案**：per-group E4M3 + per-tensor FP32 额外 scale（扩展可表示范围）

### 对比

| 维度 | MXFP4 | NVFP4 |
|------|-------|-------|
| Group size | 32 | **16** |
| Scale 格式 | E8M0（2 的幂） | **E4M3**（分数精度） |
| Scale 层级 | 一级 | **两级**（+ per-tensor FP32） |
| 性能 | 较好 | **更好** |

## 实验发现

来自 [[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 的关键发现（LLaMA-3.2-1B, WikiText-2 PPL）：

### 无预量化变换时

| 格式 | RTN | GPTQ | GPTQ+Low-rank |
|------|-----|------|---------------|
| INT4 per-channel | 256.97 | 161.04 | 139.31 |
| FP4 per-channel | **135.68** | **101.36** | **96.33** |
| MXFP4 (g32, E8M0) | 15.17 | 13.24 | 12.69 |
| NVFP4 (g16, E4M3) | **11.54** | **10.98** | **10.83** |

Per-channel 下 FP4 远优于 INT4。NVFP4 凭借更小 group size 和更精确 scale 格式显著领先 MXFP4。

### 有 Rotation 预处理后

| 格式 | Rotation+Scaling+GPTQ |
|------|----------------------|
| INT4 per-channel | **11.80** |
| FP4 per-channel | 12.13 |
| MXFP4 | 12.37 |
| NVFP4 | 11.04 |

**Rotation 后 INT4 反超 FP4**（11.80 vs 12.13）。原因：rotation 消除 outlier 使分布变均匀，而 INT4 在均匀分布上更强。

### Scaling factor 精度的影响

将 scaling factor 降到 4-bit（E2M1）会导致灾难性退化（PPL 从 11.54 暴增到 4130+），说明 scaling factor 的精度是 FP4 性能的关键瓶颈。

## 对现有方法体系的启示

FP4 的出现挑战了一些基于 INT4 建立的直觉：

1. [[Rotation-based Quantization]] 对 MXFP4/NVFP4 几乎无效——小 group size 已经起到了类似作用
2. 传统 [[Equivalent Scaling Transformation]] 仍然有效，因为 scaling 调整的是通道间平衡，与 group-wise 的 FP4 正交
3. Scaling factor 的格式和精度成为新的性能瓶颈，值得进一步研究
4. **分布匹配原则**：FP4 的优势本质来自其非均匀量化点对正态分布的适配。当 rotation 使分布变均匀后，这一优势消失甚至反转。详见 [[Quantization Insights|Insight 7]]

### 开放问题：FP4 时代需要什么预处理？

旋转对 FP4 无效是最重要的开放研究方向之一：
- 传统 outlier suppression 思路（rotation/scaling）不再适用于已有细粒度 group 的 FP4
- 可能方向：优化 group 内的数据排列？自适应 group 划分？scale factor 格式优化？

## 与已有知识的关联

- 与 [[Post-Training Quantization]] 的关系：FP4 是 PTQ 的新数据格式选项，可以复用现有的预量化变换和误差补偿方法
- 与 [[Weight-Only Quantization]] 的关系：FP4 扩展了 W4 量化从 INT4 到浮点域
- 与 [[Quantization Granularity]] 的深度交互：FP4 的行为高度依赖 group size——粗粒度下远优于 INT4，细粒度下差距消失（Insight 6）
- 硬件层面与 NVIDIA Blackwell 架构紧密绑定，是从算法到硬件 co-design 的典型案例
