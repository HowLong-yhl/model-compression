---
title: Diffusion Model Quantization
aliases: [扩散模型量化, DM Quantization]
tags: [model-compression, quantization, concept, diffusion-model, PTQ, image-generation]
created: 2026-04-23
updated: 2026-04-23
---

## 核心概念

扩散模型（Diffusion Models）通过多步去噪过程生成图像/视频，计算开销极大（单次生成需 20-50 步前向推理）。量化是加速扩散模型部署的核心技术之一，但扩散模型相比 LLM 有三个独特挑战：

1. **时间步依赖的激活分布**——激活值的分布随去噪时间步 t 剧烈变化，传统的静态量化范围无法适应
2. **误差累积效应**——每个时间步的量化误差会沿去噪链传播并放大，最终严重影响生成质量
3. **架构异质性**——UNet 中 skip connection 和上采样的特征分布差异巨大；新兴的 DiT（Diffusion Transformer）架构又带来新的量化特性

这些挑战使得直接套用 LLM 量化方法（如 [[GPTQ]]、[[AWQ]]）效果不佳，催生了专门的扩散模型量化研究。

## 方法分类

根据 [[Low-bit Model Quantization Survey]] 第 3.7 节的分类，扩散模型量化方法可从**解决的核心问题**分为四大类：

### 1. 校准集构建（Calibration Set）

扩散模型的校准需要考虑时间步分布。传统均匀采样时间步不是最优的。

| 方法 | 年份 | 会议 | 核心思路 |
|------|------|------|---------|
| **PTQ4DM** | 2023 | CVPR | 首个扩散模型 PTQ 方法；用偏态正态分布采样时间步，生成校准数据 |
| **Q-Diffusion** | 2023 | ICCV | 发现相邻时间步的激活分布相似，分组校准；对 UNet skip-connection 和上采样特征用不同量化器 |
| **TFMQ-DM** | 2024 | CVPR | 无需校准数据（training-free），利用时间步感知的通道校正 |
| **APQ-DM** | 2024 | — | 结合自适应校准与逐组量化函数，处理跨时间步的分布变化 |

**PTQ4DM** 是该领域的开创性工作。它发现扩散模型在接近噪声端（t 较大）和接近数据端（t 较小）的激活分布差异极大，因此用偏态分布（而非均匀分布）采样时间步来构建更具代表性的校准集。

### 2. 激活分布处理（Activation Distribution）

针对激活值随时间步变化的问题，多种方法提出动态量化参数。

| 方法 | 年份 | 核心思路 |
|------|------|---------|
| **Q-DM** | 2023 | 平滑激活值使其对随机时间步不敏感 + 训练中的时间步感知量化 |
| **TDQ** | 2024 | 用几何傅里叶频率编码将时间步映射为高维特征向量，再通过 MLP 映射到量化区间 |
| **BiDM** | 2024 | 观察到固定 scale factor 在所有时间步上失效，提出动态 scale |
| **APQ-DM** | 2024 | 为每个 group 设计基于激活分布变化特性的量化函数 |

### 3. 量化误差补偿（Quantization Error）

这是扩散模型量化中方法最多、进展最快的方向。因为去噪链的误差累积效应，补偿量化误差对最终生成质量至关重要。

| 方法 | 年份 | 会议 | 核心思路 | 比特 |
|------|------|------|---------|------|
| **PTQD** | 2023 | NeurIPS | 将量化噪声建模为高斯噪声，纳入去噪过程的均值和方差校正 | W8A8 |
| **BitsFusion** | 2024 | — | 用 Beta 分布采样最大误差时间步，蒸馏补偿 | W4A8 |
| **MixDQ** | 2024 | ECCV | 发现首个 token（BOS）激活幅度远大于其余 token，预计算 BOS 的全精度输出后拼接 | W4A8 混合精度 |
| **StepbaQ** | 2024 | NeurIPS | 逐时间步校准：减去均值量化误差 + 缩放修正分布偏移 + 自适应步长控制 | W4A8 |
| **MPQ-DM** | 2025 | AAAI | 基于 [[SmoothQuant]] 思想迁移激活 outlier 到权重 + 多时间步平滑蒸馏目标 | W2-W4 混合 |
| **SVDQuant** | 2025 | ICLR | 用低秩分支（SVD）吸收权重 outlier，最小化残差量化误差 | **W4A4** |
| **TAC-Diffusion** | 2024 | ECCV | 逐时间步校准；NER 模块线性缩放部分重建全精度噪声估计 | W4A8 |

**SVDQuant**（MIT Han Lab，ICLR 2025）是目前最激进的方案——实现了扩散模型的 **W4A4** 量化。其核心思路与 LLM 量化中的 outlier 处理一脉相承：先用 SVD 分解提取低秩分支来吸收权重中的 outlier（类似 [[SmoothQuant]] 将 outlier 从激活迁移到权重的思想），使残差权重的数值范围大幅缩小，从而可以安全量化到 4-bit。配套的 Nunchaku 推理引擎在 NVIDIA 4090 上实现了 FLUX.1 模型 **3.5× 加速**（12B 参数，10.1s → 2.9s）。

### 4. 新架构适配（DiT 专用方法）

随着 Diffusion Transformer（DiT）取代 UNet 成为主流架构（FLUX、Stable Diffusion 3、Sora 等），专用的量化方法随之出现。DiT 量化已成为该领域最活跃的研究方向。

#### DiT vs UNet：量化挑战的差异

DiT 相比 UNet 引入了新的量化挑战。[[ViDiT-Q]] 系统分析了 DiT 的**四重数据变异**：

| 变异维度 | DiT 特有？ | 说明 |
|---------|-----------|------|
| Token-wise | 部分（视频 DiT 更严重） | 视觉 token 间幅度差异大，视频中同时有空间和时间维度变异 |
| Condition-wise | 是 | CFG 的条件/无条件 forward 激活分布显著不同 |
| Timestep-wise | 否（UNet 也有） | 同一层在不同时间步下的激活分布变化 |
| Channel-wise (时变) | 是（比 UNet 更严重） | 通道不平衡程度随时间步变化（t=999 时 Max/Mean=40.75，t=0 时仅 12.83） |

这四重变异的叠加使得 LLM 量化方法（SmoothQuant 单一 α、QuaRot 固定旋转）和早期扩散模型方法（Q-Diffusion 粗粒度参数）在 DiT 上效果大打折扣，尤其在文本引导生成的 W4A8 设定下。

#### DiT 量化方法总览

| 方法 | 年份 | 会议 | 目标 | 最低精度 | 核心技术 | 文本引导效果 |
|------|------|------|------|---------|---------|------------|
| **PTQ4DiT** | 2024 | NeurIPS | 类别条件 DiT | W8A8 | 固定 saliency mask + channel 重参数化（类 SmoothQuant） | 未验证 |
| **Q-DiT** | 2025 | CVPR | DiT (PixArt) | W4A8* | Channel-wise grouping + 进化搜索 group size | W4A8 失败 |
| **[[ViDiT-Q]]** | 2025 | ICLR | DiT + Video DiT | **W4A8** | Static-Dynamic channel balance + Metric-Decoupled 混合精度 | **近无损** |
| **SVDQuant** | 2025 | ICLR | DiT (FLUX) | **W4A4** | SVD 低秩分支吸收 outlier + Nunchaku 引擎 | FLUX W4A4 3.5× 加速 |
| **Q-VDiT** | 2025 | ICML | Video DiT | **W3A6** | 量化 + 蒸馏联合优化 | 视频 DiT 近无损 |
| **TQ-DiT** | 2025 | — | DiT | W4A8 | 时间步感知的量化参数动态调整 | — |
| **TaQ-DiT** | 2024 | — | DiT | W8A8 | 时间步感知量化 | — |

*Q-DiT 在 class-conditioned 上 W4A8 可用，但文本引导生成上完全失败（VQA≈0）。

#### 技术路线对比

DiT 量化形成了三条主要技术路线：

**路线一：Channel Balance（ViDiT-Q、PTQ4DiT）** —— 核心思路是处理 DiT 的通道不平衡。ViDiT-Q 的 Static-Dynamic 方案最为完整：将静态缩放（基于 t≈999 最坏情况计算的固定 s）和正交旋转（固定 Hadamard Q）**同时串联**应用于所有时间步——缩放解决预训练遗留的极端不均衡，旋转自适应处理剩余的时变波动，两者配合将 [[Incoherence]] 在所有时间步都降至约 5。

**路线二：Outlier Absorption（SVDQuant）** —— 来自 [[MIT Han Lab]] 的方案，用 SVD 低秩分支吸收权重 outlier（类似 [[SmoothQuant]] 的 outlier 迁移思想），实现了最激进的 **W4A4** 量化。配套的 Nunchaku 引擎实现实际部署加速。

**路线三：量化 + 蒸馏（Q-VDiT）** —— ICML 2025 的最新工作，将量化与知识蒸馏联合优化，实现了 W3A6 的极低比特视频 DiT 量化。

**PTQ4DiT**（NeurIPS 2024）发现 DiT 架构中激活 outlier 的分布具有**时间步依赖性**——不同时间步的 outlier channel 位置和幅度不同，这比 LLM 中固定的 outlier channel 更难处理。它提出了基于 channel-wise 重参数化（类似 [[Equivalent Scaling Transformation]]）的自适应量化方案，但仅在类别条件生成上验证。

**ViDiT-Q**（ICLR 2025）在 PTQ4DiT 的基础上提出了关键改进：不仅发现旋转后仍有残余不平衡（incoherence 12.83 对 4-bit 仍然太大），还发现仅优化量化误差（MSE）不足以保障视觉生成质量——需要针对不同层类型使用解耦的评价指标（文本对齐用 ClipScore、视觉质量用 ImageReward、时间一致性用 CLIP-temp）。

## 与 LLM 量化的关系

扩散模型量化与 LLM 量化共享许多核心技术，但针对不同的瓶颈：

| 维度 | LLM 量化 | 扩散模型量化 |
|------|---------|------------|
| 核心瓶颈 | 固定 channel 上的 [[Emergent Outlier Features\|outlier]] | 时间步依赖的动态激活分布 |
| 主流架构 | Transformer decoder | UNet → DiT (Transformer) |
| 预量化变换 | Rotation / Scaling（[[Rotation-based Quantization\|旋转]]、[[Equivalent Scaling Transformation\|缩放]]） | Smooth（类似 Scaling）、SVD 低秩分解 |
| 误差补偿 | GPTQ（一次性） | 需要沿去噪链逐步补偿 |
| 关键精度 | W4A4 已实用 | W4A8 为主流，W4A4 刚由 SVDQuant 突破 |
| 技术交叉 | [[SmoothQuant]]、[[AWQ]] | MPQ-DM 直接使用 SmoothQuant；SVDQuant 来自 [[MIT Han Lab]]，与 AWQ 同源 |

值得注意的是 SVDQuant 的作者团队（MIT Han Lab）同时也是 [[AWQ]] 和 [[SmoothQuant]] 的提出者，LLM 量化的 outlier 处理经验直接迁移到了扩散模型场景。

## 研究时间线

```
2023.02  PTQ4DM (CVPR 2023)       ← 首个扩散模型 PTQ
2023.02  Q-Diffusion (ICCV 2023)  ← skip-connection 分组量化
2023.05  PTQD (NeurIPS 2023)      ← 量化噪声建模
2023.11  Q-DM (NeurIPS 2023)      ← 激活平滑 + 时间步感知 QAT
2024.05  PTQ4DiT (NeurIPS 2024)   ← 首个 DiT 量化方法
2024.05  MixDQ (ECCV 2024)        ← 混合精度 + BOS token 处理
2024.06  ViDiT-Q (ICLR 2025)      ← 首个 text-to-image/video DiT 量化，Static-Dynamic Balance
2024.10  StepbaQ (NeurIPS 2024)   ← 逐时间步校准
2024.11  SVDQuant (ICLR 2025)     ← W4A4 突破，Nunchaku 引擎 3.5× 加速
2025.01  Q-DiT (CVPR 2025)        ← DiT channel-wise grouping + 进化搜索
2025.01  MPQ-DM (AAAI 2025)       ← 极低比特混合精度
2025.05  Q-VDiT (ICML 2025)       ← 视频 DiT 量化 + 蒸馏，W3A6
```

## 开放问题

1. **W4A4 在扩散模型上的质量**: SVDQuant 实现了 W4A4，但生成质量（FID/CLIP Score）相比 LLM 的 PPL 指标更敏感——视觉伪影比文本错误更容易被人类察觉。ViDiT-Q 的 Static-Dynamic Balance 能否支撑 W4A4 尚未验证
2. **视频扩散模型**: 视频 DiT（Sora、CogVideoX）的参数量远超图像模型，量化需求更迫切，但时空注意力的量化难度更大。Q-VDiT（ICML 2025）开始探索这一方向
3. **评价指标的不一致性**: ViDiT-Q 揭示了一个根本问题——MSE 与人类感知的视觉质量之间存在严重偏差。如何设计更好的量化感知评价指标是开放问题
4. **与 Rotation-based 方法的深度融合**: ViDiT-Q 证明旋转 + 可学习 scaling 的组合有效，但尚未探索 SpinQuant 式的学习最优旋转在 DiT 上的效果
5. **Attention 计算的量化**: 当前方法（ViDiT-Q、Q-DiT）仅量化 Linear 层，未触及 Attention 的 QK 矩阵乘法（占 ~14% 延迟），这部分的量化仍是空白

## 引用关系

← 综述来源：[[Low-bit Model Quantization Survey]]（第 3.7 节系统分类）、[[Diffusion Model Quantization - A Review]]（浙大，首篇专门综述，统一基准实验 + 伪影分析）
← 技术关联：[[SmoothQuant]]（outlier 迁移思想 → MPQ-DM）、[[AWQ]]（MIT Han Lab → SVDQuant）
← LLM 量化对应概念：[[Emergent Outlier Features]]（扩散模型有时间步依赖的类似现象）、[[Equivalent Scaling Transformation]]（PTQ4DiT 的重参数化）
→ 代表方法：PTQ4DM、Q-Diffusion、SVDQuant、PTQ4DiT、MixDQ、[[ViDiT-Q]]、Q-VDiT
→ 硬件加速：Nunchaku（SVDQuant 配套推理引擎，类似 LLM 中的 TinyChat / [[bitsandbytes]]）
