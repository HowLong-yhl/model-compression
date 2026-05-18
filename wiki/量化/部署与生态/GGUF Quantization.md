---
title: GGUF Quantization
aliases: [GGUF 量化, GGUF 格式, llama.cpp 量化]
tags: [model-compression, quantization, concept, deployment, GGUF]
created: 2026-04-30
updated: 2026-04-30
sources: []
---

## 定义

GGUF（GPT-Generated Unified Format）是 llama.cpp 生态的模型量化与推理格式。不同于 GPTQ/AWQ 等仅关注量化算法本身，GGUF 是一个完整的**生态系统**——包含量化算法、文件格式和推理引擎，支持 CPU、GPU 和混合推理。

本页基于 [gguf-docs](https://github.com/iuliaturc/gguf-docs) 非官方文档整理。

## 生态架构

```
GGML（tensor library）
  └── llama.cpp（推理引擎）
        └── GGUF（文件格式 + 量化算法）
```

- **GGML**：底层 tensor 库，实现矩阵运算，支持 CPU SIMD（AVX2/AVX-512/ARM NEON）和 GPU offload（CUDA/Metal/Vulkan/ROCm）
- **llama.cpp**：基于 GGML 的 LLM 推理引擎，提供量化工具链
- **GGUF**：二进制文件格式，存储量化后的模型权重和元数据

推理生态：Ollama、LM Studio、koboldcpp、text-generation-webui 均基于 llama.cpp。

## 三代量化算法

GGUF 的量化算法经历了三代演进，每一代都在精度-压缩率权衡上取得显著改进。

### 第一代：Legacy Quants（遗留量化）

最早的量化实现，使用**标量 block 量化**——将权重分成固定大小的 block，每个 block 共享量化常数。

#### Type 0：对称量化

代表类型：`Q4_0`、`Q5_0`、`Q8_0`

将权重以零为中心对称量化。每个 block 只需存储 1 个 FP16 scale 常数 `S`：

```
量化：quant(w) = round(w / S)
反量化：dequant(q) = q × S

其中 S = max(|w|) / max_int
```

- Block size：32 个权重
- 每 block 存储开销：1 个 FP16（2 bytes）
- INT4 的特殊处理：4 bit 有 16 个值（-8 到 +7），但丢弃最左 bin（-8），保留 15 个值，使零点两侧 bin 数相等

**缺点**：对称量化假设权重分布以零为中心。当分布不对称时，量化范围的一半被浪费。

#### Type 1：非对称量化

代表类型：`Q4_1`、`Q5_1`、`Q8_1`

引入 zero-point offset，允许量化范围偏移以适配非对称分布。每个 block 存储 2 个 FP16 常数——scale `S` 和 offset `α`（min value）：

```
量化：quant(w) = Z + round(w / S)
反量化：dequant(q) = (q - Z) × S + α

其中 S = (max(w) - min(w)) / (max_int - min_int)
     Z = -round(min(w) / S)
```

- Block size：32 个权重
- 每 block 存储开销：2 个 FP16（4 bytes）

**权衡**：更精确（尤其对非对称分布），但量化常数存储翻倍。对 16B 参数模型，Type 1 的常数开销约 **2 GB**。

### 第二代：K-Quants

代表类型：`Q2_K`、`Q3_K_S/M/L`、`Q4_K_S/M`、`Q5_K_S/M`、`Q6_K`

"K" 最可能代表开发者 **Kawrakow** 的名字（而非 K-means），也可能代表 "kernel"。K-Quants 引入了三项关键创新。

#### 创新一：Super-Block + Double Quantization

核心问题：Legacy quants 中，每个 block 的量化常数（scale、offset）存储为 FP16，开销显著。

解决方案：**对量化常数本身再做量化**（double quantization）。

```
Super-Block 结构：
┌─────────────────────────────────────────────────┐
│ Super-Block                                      │
│ ┌──────┐ ┌──────┐ ┌──────┐     ┌──────┐        │
│ │Block₁│ │Block₂│ │Block₃│ ... │Block₈│  ×8    │
│ │32 wts│ │32 wts│ │32 wts│     │32 wts│        │
│ └──┬───┘ └──┬───┘ └──┬───┘     └──┬───┘        │
│    │        │        │             │             │
│    S₁(INT8) S₂(INT8) S₃(INT8)    S₈(INT8)      │ ← block constants 量化为 INT8
│                                                  │
│    S_super(FP16)  +  α_super(FP16)              │ ← super-block constants 保持 FP16
└─────────────────────────────────────────────────┘
  共 256 个权重 (8 blocks × 32 weights)
```

- 8 个 regular block 组成 1 个 super-block
- 每个 regular block 的 scale/offset 量化为 **INT8**
- 每个 super-block 保留 1 对 **FP16** 常数用于反量化 block constants
- 存储节省：对 16B 参数模型，常数开销从 ~2 GB 降至 ~1 GB

额外好处：super-block 结构改善了内存访问的连续性，提高了 cache line 利用率。

#### 创新二：Mixed Precision（混合精度）

不同类型的层对量化的敏感度不同。K-Quants 为不同层分配不同精度：

| 层类型 | 精度分配 | 理由 |
|--------|----------|------|
| LayerNorm 权重 | FP16（不量化） | 参数极少但对输出影响大 |
| Attention 投影 | 较高精度（如 Q5_0、Q6_K） | 注意力模式对精度敏感 |
| FFN 层 | 主要量化目标（如 Q4_K） | 参数最多，容忍量化较好 |
| Output 层 | 较高精度 | 直接影响 logits 输出 |

#### 创新三：Size Modifier（S/M/L）

同一比特宽度下，通过 **size modifier** 控制混合精度的分配策略：

| Modifier | 含义 | 策略 |
|----------|------|------|
| S（Small） | 最小文件 | 非主要层用更低精度 |
| M（Medium） | 平衡 | **最常用**（如 Q4_K_M） |
| L（Large） | 最高精度 | 非主要层也用较高精度 |

Q4_K_M 是 K-Quants 中最被广泛推荐的通用量化等级。

### 第三代：I-Quants

代表类型：`IQ1_S/M`、`IQ2_XXS/XS/S/M`、`IQ3_XXS/XS/S`、`IQ4_XS/NL`

"I" 代表 **Importance**（importance matrix）。I-Quants 从标量量化跃迁到**向量量化**，在极低比特率（1-3 bpw）下实现了质的飞跃。开发者 Kawrakow 引用 [[QuIP#]] 论文为灵感来源。

#### 核心：向量量化 + Codebook

K-Quants 独立量化每个权重（标量量化），I-Quants 将 **8 个权重视为一个不可分割的 8 维向量**，在预定义的 codebook 中查找最近邻：

```
标量量化（K-Quants）：     向量量化（I-Quants）：
w₁ → q₁                  [w₁,w₂,...,w₈] → codebook[idx]
w₂ → q₂                  一个 8-bit code 编码 8 个权重
...
w₈ → q₈
```

- **Codebook**：硬编码在 `ggml-quants.h` 中的参考向量集，编码为十六进制值
- **查找**：量化时在 codebook 中搜索与输入向量最近的原型

#### Sign Trick（符号技巧）

关键优化：codebook 只存储**正向量**。量化时：

1. 取输入向量的绝对值
2. 在 codebook 中搜索最近邻 → 得到 8-bit code
3. 将 8 个维度的符号打包成 **1 byte**（8 个 sign bits）

效果：

- 存储的 codebook 大小 = 256 条（8-bit code）
- 等效 codebook 大小 = 256 × 2⁸ = **65,536 条**（256× 扩展）
- 每个向量的存储 = 8-bit code + 8-bit signs = **2 bytes 编码 8 个权重**

#### Scale 共享

FP16 scale 值在多个向量间共享：1 个 FP16 scale 覆盖 **256 个向量**（= 2048 个权重），仅贡献 ~0.008 bpw 的额外开销。

#### IQ2 存储计算

以 IQ2 为例（~2 bpw）：

| 分量 | 每权重比特 |
|------|-----------|
| 8-bit code（8 权重） | 1.0 bpw |
| 8-bit signs（8 权重） | 1.0 bpw |
| FP16 scale（2048 权重） | ~0.008 bpw |
| **合计** | **~2.008 bpw** |

## Importance Matrix（imatrix）

imatrix 是 I-Quants 的质量关键——通过 calibration 数据计算每个权重的重要性，指导量化过程优先保护重要权重。

### 计算过程

1. **Calibration**：在代表性文本（通常几百个 WikiText 样本）上运行推理
2. **Per-row 重要性**：对每一行 i，计算 `I_i = y_i²`（输出激活的平方）
3. **Per-weight 重要性**：结合激活和权重本身 `I_ij = y_i² + √(σ² + w_ij²)`

### 去耦化技巧（Decoupled Constants）

imatrix 的精妙之处在于**量化常数和去量化常数可以不同**：

```
量化阶段：使用 (S, Z) 将权重映射到量化点
去量化阶段：使用 (S', Z') ≠ (S, Z) 将量化点映射回浮点

S' 和 Z' 通过加权最小二乘法求解，权重为 importance score
→ 去量化自动偏向重要权重的精确重建
```

具体实现：对 `S'` 进行 36 值 grid search，选择使 importance-weighted 重建误差最小的值。

### 开销

- **推理时**：零额外开销（imatrix 只影响量化过程，不影响推理）
- **量化时**：增加 10-20% 的量化时间
- **存储**：极少量额外存储（去量化常数本身的调整）

## 命名规范

GGUF 量化类型的命名遵循 `[算法版本][比特宽度]_[size modifier]` 的结构：

```
Q4_K_M
│ │ │ └── Size modifier: S(small)/M(medium)/L(large)
│ │ └──── 算法代: K=K-Quants, 无后缀=Legacy
│ └────── 比特宽度: 4-bit
└──────── Q = Quantized

IQ2_XS
│  │  └── Size modifier: XXS/XS/S/M
│  └───── 比特宽度: 2-bit
└──────── IQ = Importance-matrix Quantization
```

常见类型速查：

| 类型 | 算法代 | 近似 bpw | 常用场景 |
|------|--------|----------|---------|
| Q4_0 | Legacy | ~4.5 | 基线，已被 Q4_K_M 取代 |
| Q4_K_M | K-Quants | ~4.58 | **最常用通用量化** |
| Q5_K_M | K-Quants | ~5.52 | 高质量，接近无损 |
| Q8_0 | Legacy | ~8.50 | 本质无损 |
| IQ2_M | I-Quants | ~2.70 | 大模型（70B+）极限压缩 |
| IQ4_XS | I-Quants | ~4.25 | 与 Q4_K_S 竞争，bpw 略低 |

## 量化工作流

```bash
# Step 1: HuggingFace 模型转换为 GGUF（FP16/BF16）
python convert_hf_to_gguf.py /path/to/model --outfile model-f16.gguf

# Step 2: 量化
# K-Quants（无需 calibration）
llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M

# I-Quants（需先计算 imatrix）
llama-imatrix -m model-f16.gguf -f calibration-data.txt -o imatrix.dat
llama-quantize --imatrix imatrix.dat model-f16.gguf model-IQ2_M.gguf IQ2_M
```

## 三代算法对比

| 维度 | Legacy Quants | K-Quants | I-Quants |
|------|--------------|----------|----------|
| 量化方式 | 标量 block | 标量 super-block | **向量 codebook** |
| 量化常数优化 | 无（直接 FP16） | **double quantization** | 去耦化 + grid search |
| 混合精度 | 无 | **有**（LayerNorm FP16） | 有 |
| 需要 calibration | 否 | 否 | **是**（imatrix） |
| 极低 bit 质量 | 差 | 中 | **好** |
| 4-bit 质量 | 可用 | **好** | 好（略优） |
| 理论基础 | 标量均匀量化 | 分层量化 | [[QuIP#]] 启发的 VQ |

## 与 wiki 其他内容的关联

- **与 [[Quantization Deployment Formats]] 的关系**：GGUF 是本页深入的技术细节，Deployment Formats 提供格式间的对比和选型框架
- **与 [[QuIP#]] / [[Lattice Codebooks]] 的关系**：I-Quants 的 codebook VQ 方法受 QuIP# 启发，但未使用 E₈ 格码本，而是硬编码的自定义 codebook
- **与 [[Weight-Only Quantization]] 的关系**：GGUF 本质是 weight-only PTQ（权重量化 + FP16/BF16 激活），通过 CPU SIMD 和 GPU offload 加速去量化计算
- **与 [[Post-Training Quantization]] 两步分解框架的映射**：Legacy/K-Quants 对应 Step 2 RTN（直接取整）；I-Quants 的 imatrix 引入了数据依赖的量化优化，但不属于传统的 Hessian 补偿路线
- **与 [[Model Quantization Ecosystem]] 的关系**：GGUF 是 Qwen（官方提供）、DeepSeek（社区 bartowski/unsloth）、Gemma（社区 + QAT 转换）三大系列的核心本地推理格式
