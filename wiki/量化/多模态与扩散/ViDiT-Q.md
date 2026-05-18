---
title: "ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation"
aliases: [ViDiT-Q]
source_type: paper
source_url: "https://arxiv.org/abs/2406.02540"
source_date: 2024-06-04
ingested: 2026-04-23
venue: ICLR 2025
authors: [Tianchen Zhao, Tongcheng Fang, Haofeng Huang, Rui Wan, Widyadewi Soedarmadji, Enshu Liu, Shiyao Li, Zinan Lin, Guohao Dai, Shengen Yan, Huazhong Yang, Xuefei Ning, Yu Wang]
affiliation: Tsinghua University / InfinigenceAI / Microsoft / SJTU
code: "https://github.com/A-suozhang/ViDiT-Q"
tags: [model-compression, quantization, method, post-training, diffusion-model, DiT, video-generation, mixed-precision]
---

## 核心要点

ViDiT-Q 是首个系统解决 **文本到图像/视频** DiT 量化的方法。此前的 DiT 量化工作（PTQ4DiT、Q-DiT）仅在类别条件生成（class-conditioned）上验证，直接应用于更具挑战性的文本引导生成时会严重失败（W4A8 下生成空白图像或纯噪声）。ViDiT-Q 通过三组针对性技术，在 OpenSORA 视频生成和 PixArt-α 图像生成上实现了 W4A8 近无损量化，配合高效 CUDA kernel 达到 2-2.5× 内存节省和 1.4-1.7× 延迟加速。

## DiT 量化的独特挑战

ViDiT-Q 的核心贡献之一是系统分析了 DiT 相比传统 UNet 扩散模型和 LLM 的**四重数据变异**：

| 变异维度 | 现象 | 影响 |
|---------|------|------|
| **Token-wise** | 视觉 token 间幅度差异大；视频 DiT 中同时存在空间和时间维度变异 | 粗粒度（tensor-wise）量化参数无法适应 |
| **Condition-wise** | CFG（Classifier-Free Guidance）的条件/无条件两个 forward 的激活分布差异显著 | 静态量化参数不够 |
| **Timestep-wise** | 同一层在不同去噪时间步 t 下激活分布剧烈变化 | 固定 scale factor 失效 |
| **Channel-wise (时变)** | 通道不平衡程度随时间步变化——早期 t=999 时 Max/Mean=40.75，晚期 t=0 时仅 12.83 | 单一 α 的 scaling 或固定旋转不够 |

这四重变异的叠加使得现有方法失败：Q-Diffusion/PTQ4DM 的粗粒度静态参数无法应对 token/condition 变异；PTQ4DiT 的固定 channel balance mask 无法适应时变通道不平衡；Q-DiT 的 channel-wise grouping 在 W4 下仍不够精细；SmoothQuant/QuaRot 等 LLM 方法未考虑时间步维度。

## 方法详解

### 技术一：细粒度分组 + 动态量化参数（Sec 4.1）

将量化参数从 **静态 tensor-wise** 升级为 **动态 token-wise**：

- **权重**：per-channel 量化（标准做法）
- **激活**：per-token 动态量化——每个 token 独立计算 scale/zero-point

这一简单改动将 VQA 从近乎零分提升至可读水平。关键观察：硬件上 token-wise 激活量化的开销可忽略（量化参数沿 input-channel 维度共享，与 GEMM 兼容）。

### 技术二：Static-Dynamic Channel Balance（Sec 4.2）

DiT 的通道不平衡有两个**来源**（static vs dynamic），ViDiT-Q 用两种变换**同时串联**在所有时间步上处理，而非分阶段切换：

```
所有时间步 t ∈ [T, 0] 都执行：
    X' = X · diag(s)⁻¹     ← 静态缩放（固定 s，解决预训练遗留的极端通道不均衡）
    X'' = X' · Q            ← 正交旋转（固定 Q，自适应处理剩余的时变分布差异）
    然后对 X'' 进行动态量化（实时计算当前 t 的 scale/zero-point）
```

两种变换都是**线性操作**，可融合为一个等效变换矩阵 `diag(s)⁻¹ · Q` 永久吸收到权重中。推理时无需"切换"——它们各司其职但同时生效。

**Static Scaling——解决"历史性"问题**

职责：处理预训练模型遗留的**静态通道不平衡**（主要来自 DiT 的 Scale Shift Table 机制）。

实现：基于 **t≈999**（最极端的初始去噪阶段）的激活统计计算固定的 per-channel 缩放因子 s，然后**应用于所有时间步**。针对最坏情况设计的 s 在较温和的 t<T 时只是略微"过度缩放"，但不会造成灾难性后果——后续的旋转会修正这种过度。

**Rotation——解决"动态性"问题**

职责：处理时间嵌入 γ(t), β(t) 引入的**时变分布变化**。

实现：使用固定的正交矩阵 Q（如 Hadamard），**也应用于所有时间步**。"动态"不是指 Q 随时间变化，而是正交变换的**天然自适应性**——无论当前是 t=999 还是 t=0，旋转都能将通道间的相关性打散到各维度。当固定 s 在某些时间步不是最优选择时，旋转修正剩余的不均衡。

**"各司其职"的分工体现**：

| 时间步 | 原始 Incoherence | 缩放后 | 缩放+旋转后 | 说明 |
|--------|-----------------|--------|------------|------|
| t=999 | 40.75 | ~17.77 | **~5** | 缩放贡献最大（40→17），旋转进一步优化（17→5） |
| t=0 | ~12.83 | ~8 | **~5** | 缩放贡献较小（12→8），旋转拉平残余（8→5） |

关键观察：在**所有时间步都达到约 5** 的低 [[Incoherence]]，而非各自为政时的 12-40。

**与现有方法的对比**：

| 方法 | Channel Balance 策略 | 局限 |
|------|---------------------|------|
| [[SmoothQuant]] | 单一 α 的 channel scaling | 单一 α 无法适应不同时间步的 imbalance 程度 |
| [[QuaRot]] | 全局 Hadamard 旋转 | Incoherence 降至 12.83 但对 4-bit（16 级）仍太大 |
| PTQ4DiT | 固定 saliency mask | 固定 mask 无法适应跨时间步的变化 |
| **ViDiT-Q** | **缩放 + 旋转串联，全时间步生效** | 缩放当"保险"，旋转当"修理工" |

### 技术三：Metric-Decoupled Mixed Precision（Sec 4.3）

ViDiT-Q 发现量化对生成质量的影响与**层类型高度相关**：

- **Cross-Attention 层**：主要影响**文本对齐**（text alignment），对视觉质量影响小
- **Self-Attention + FFN 层**：主要影响**视觉保真度**（fidelity），对文本对齐影响小
- **Temporal Attention 层**（仅视频）：主要影响**时间一致性**

基于此发现，ViDiT-Q 设计了"解耦"的混合精度分配策略：

1. 按 MSE 对每个 group 做初始 bit-width 预算分配
2. 按层类型分组，用对应的评价指标（ClipScore / ImageReward / CLIP-temp）做层级敏感度分析
3. 迭代式将最敏感的层提升到更高 bit-width，直到满足总参数预算

这比传统 MSE-only 的混合精度显著更好——MSE 倾向于过度保护"数值变化大"的层（如 cross-attention），而忽略对视觉质量真正重要的层。

## 量化范围

ViDiT-Q 仅量化 **Linear 层**（GEMM 操作），不量化 Attention 的 Softmax / 点积等非线性计算。这一设计选择的核心原因是：现代推理框架普遍采用 **Flash Attention**，Attention 计算本身已高度优化且对总延迟的贡献有限，真正的计算瓶颈在于参数量和 FLOPs 占比压倒性的 Linear 层。

### 被量化的层（INT4 / INT8）

| 层类型 | 具体矩阵 | 说明 |
|--------|----------|------|
| Q/K/V 投影 | $W_q, W_k, W_v$ | Self-Attention 和 Cross-Attention 中的线性投影 |
| Attention 输出投影 | $W_o$ | Attention 后的线性映射 |
| MLP（FFN） | $W_{fc1}, W_{fc2}$ | 前馈网络的两层线性变换 |

这些层的特点是：**计算密集型**，占模型 FLOPs 的绝大部分，量化收益最大。

### 保持 FP16 的层

| 层 / 操作 | 原因 |
|-----------|------|
| Attention Softmax | 数值稳定性要求高，低精度易导致梯度消失 / 爆炸 |
| Attention 点积 $QK^T$ | 需要高精度避免数值溢出 / 下溢 |
| LayerNorm / RMSNorm | 统计量计算（mean / var）对精度敏感 |
| AdaLN / Feature Modulation | $\gamma, \beta$ 的参数和计算保持 FP16 |
| 激活函数（GELU / SiLU 等） | 非线性操作通常保持高精度 |

## 实验结果

### 文本到视频：OpenSORA（VBench）

| 方法 | Bit (W/A) | VQA-Aesthetic | VQA-Technical | CLIPSIM | 状态 |
|------|-----------|--------------|---------------|---------|------|
| FP16 | 16/16 | 64.20 | 51.90 | 0.180 | 基线 |
| Q-DiT | 4/8 | 0.007 | 0.018 | 0.169 | **完全失败** |
| PTQ4DiT | 4/8 | 2.21 | 0.318 | 0.174 | **完全失败** |
| SmoothQuant | 4/8 | 31.96 | 22.85 | 0.183 | 严重退化 |
| QuaRot | 4/8 | 47.36 | 33.13 | 0.182 | 明显退化 |
| **ViDiT-Q** | **4/8** | **60.62** | **49.38** | **0.181** | **近无损** |

Q-DiT 和 PTQ4DiT 在 W4A8 下完全崩溃（VQA 接近零，生成空白或噪声）。即使 LLM 量化方法（SmoothQuant、QuaRot）也有严重退化。ViDiT-Q 是唯一在 W4A8 下保持可用质量的方法。

### 文本到图像：PixArt-α（COCO）

| 方法 | Bit (W/A) | FID ↓ | CLIP ↑ | ImageReward ↑ |
|------|-----------|-------|--------|---------------|
| FP16 | 16/16 | 73.34 | 0.258 | 0.901 |
| Q-DiT | 4/8 | 475.8 | 0.127 | -2.277 |
| PTQ4DiT | 4/8 | 171.9 | 0.177 | -2.064 |
| **ViDiT-Q** | **4/8** | **74.33** | **0.257** | **0.887** |

### 硬件加速

| 设定 | 内存节省 | 延迟加速 |
|------|---------|---------|
| W8A8 | 1.99× | 1.71× |
| W4A8 | 2.42× | 1.38× |

通过 kernel fusion（量化操作 + Hadamard 融入 LayerNorm/GeLU/Residual），实际部署开销极小。

## 关键 Insights

1. **仅降低量化误差不够——视觉生成需要"质量解耦"的评价**：MSE 与人类感知的视觉质量、文本对齐、时间一致性之间存在严重偏差。这是 DiT 量化与 LLM 量化的根本区别——LLM 的 PPL 是相对统一的指标，而视觉生成有多个正交的质量维度
2. **旋转是必要但不充分的**：QuaRot 式的 Hadamard 旋转将 [[Incoherence]] 从 40.75 降至 12.83，但对 4-bit 仍不够。需要额外的可学习 scaling 进一步降至 5.02。这与 LLM 量化中 [[Quantization Insights|Insight 1-2]] 的"互补组合"原则一致
3. **DiT 的时变 channel imbalance 是独特挑战**：LLM 的 [[Emergent Outlier Features]] 是固定的，但 DiT 的 outlier 分布随时间步剧烈变化——同一通道在 t=999 时是 outlier，在 t=0 时可能完全正常

## 与其他 DiT 量化方法的对比

| 方法 | 会议 | 目标架构 | 最低精度 | 核心技术 | 文本引导生成 |
|------|------|---------|---------|---------|------------|
| **PTQ4DiT** | NeurIPS 2024 | DiT | W8A8 | 固定 saliency mask + channel 重参数化 | 未验证 |
| **Q-DiT** | CVPR 2025 | DiT | W4A8* | Channel-wise grouping + 进化搜索 | W4A8 失败 |
| **ViDiT-Q** | ICLR 2025 | DiT + Video DiT | **W4A8** | Static-Dynamic balance + Metric-Decoupled MP | **近无损** |
| **SVDQuant** | ICLR 2025 | DiT (FLUX) | **W4A4** | SVD 低秩分支吸收 outlier + Nunchaku 引擎 | FLUX W4A4 |
| **Q-VDiT** | ICML 2025 | Video DiT | **W3A6** | 量化 + 蒸馏联合优化 | 视频 DiT |
| **TQ-DiT** | 2025 | DiT | W4A8 | 时间步感知的量化参数动态调整 | — |

*Q-DiT 论文中 W4A8 结果在 class-conditioned 上可用，但文本引导生成上失败。

## 局限性

- 混合精度分配需要逐层评估多个指标（ClipScore、ImageReward 等），校准开销较大
- 仅量化 Linear 层（Flash Attention 已使 Attention 计算不再是瓶颈，但 Attention 点积 / Softmax 的低精度量化仍是潜在的进一步加速方向）
- W4A4 未探索——ViDiT-Q 的 Static-Dynamic Balance 能否支撑更低精度尚不清楚
- 可学习的 scale-shift 表需要校准数据和梯度优化，不如 QuaRot 的零数据方案轻量

## 引用关系

← 技术来源：[[SmoothQuant]]（channel scaling 思想）、[[QuaRot]]（Hadamard 旋转用于 static balance）、[[Incoherence]]（显式引用 QuIP# 的 incoherence 概念）
← 对比方法：PTQ4DiT、Q-DiT、Q-Diffusion、[[SmoothQuant]]、[[QuaRot]]
← 应用场景：[[Diffusion Model Quantization]]（DiT 专用方法的代表）
← 综述收录：[[Diffusion Model Quantization - A Review]]（归入 DiT PTQ - Channel Equalization 类别）
→ 同期工作：SVDQuant（ICLR 2025，W4A4 路线）、Q-VDiT（ICML 2025，视频蒸馏路线）
→ 前序工作（同团队）：MixDQ（ECCV 2024，ViDiT-Q 第一作者的前期工作，UNet 混合精度）
→ 关键概念：[[Incoherence]]、[[Rotation-based Quantization]]、[[Equivalent Scaling Transformation]]
