---
title: "Diffusion Model Quantization: A Review"
aliases: [DM Quantization Review, 扩散模型量化综述(浙大)]
source_type: survey
source_url: "https://arxiv.org/abs/2505.xxxxx"
source_date: 2025-05-08
ingested: 2026-04-28
venue: Preprint (under review)
authors: [Qian Zeng, Chenggong Hu, Mingli Song, Jie Song]
affiliation: Zhejiang University
code: "https://github.com/TaylorJocelyn/Diffusion-Model-Quantization"
tags: [model-compression, quantization, survey, diffusion-model, PTQ, QAT, UNet, DiT, benchmark]
---

## 核心定位

这是**首篇**专门聚焦扩散模型量化的综合性 survey（40 页），系统覆盖截至 2025 年 3 月的所有相关工作。相比 [[Low-bit Model Quantization Survey]] 仅用 3.7 节（约 2 页）概述扩散模型量化，本文提供了远更深入的分析：六大挑战定义、完整方法分类体系、统一基准实验、以及量化伪影的定性分析。

作者团队同时也是 D2-DPM（量化误差校正方法）的提出者，对该领域有第一手研究经验。

## 六大挑战

本文首次将扩散模型量化的挑战系统化为六条，分属两大来源：

### 来自扩散机制的挑战

| 编号 | 挑战 | 说明 |
|------|------|------|
| C#1 | **激活分布随时间步变化** | 去噪过程中 $x_t$ 的分布随 $t$ 剧烈变化，层激活值在不同时间步差异显著 |
| C#2 | **量化误差跨时间步累积** | 每步的量化噪声 $\Delta\epsilon_\theta$ 沿去噪链传播并放大，最终采样轨迹偏离 |
| C#3 | **时间信息混淆** | TimeEmbedding 量化后，$\hat{emb}_t$ 偏向邻近时间步 $emb_{t+\delta_t}$，导致去噪轨迹回溯或迂回 |

### 来自模型架构的挑战

| 编号 | 挑战 | 架构 | 说明 |
|------|------|------|------|
| C#4 | **双峰数据分布** | UNet | Skip connection 的 concat 操作将浅层和深层特征直接拼接，产生双峰分布 |
| C#5 | **文本-图像一致性** | Text-to-Image | Cross-attention 量化影响语义对齐；`<start>` token 的注意力分布需特殊处理 |
| C#6 | **Transformer 块内数据异方差** | DiT | Transformer 块内的 attention 和 FFN 激活方差差异大（继承自 LLM 的 outlier 问题） |

与 wiki 中 [[Diffusion Model Quantization]] 页面已有的三大挑战（时间步依赖分布、误差累积、架构异质性）相比，本文新增了 C#3（时间信息混淆）和 C#5（文本-图像一致性）的独立定义，并将架构挑战细分为 UNet（C#4）和 DiT（C#6）两类。

## 方法分类体系

本文按**架构（UNet / DiT）** × **策略（PTQ / QAT）**的两级结构组织，每级再按核心技术细分：

### UNet-Based PTQ（6 大技术类别）

| 技术类别 | 解决的挑战 | 代表方法 | 核心思路 |
|---------|----------|---------|---------|
| **校准策略定制** | C#1 | PTQ4DM, EDA-DM | 设计时间步感知的校准集采样策略 |
| **双峰分布消除** | C#4 | Q-Diffusion | 在 concat 前分别量化，避免混合分布 |
| **时间动态量化** | C#1 | TDQ, APQ-DM | 用时间步编码生成动态量化参数（推理时可预计算，零额外开销） |
| **时间信息维护** | C#3 | TFMQ-DM, QVD | 独立校准 TimeEmbedding 模块，保持时间特征可区分性 |
| **量化误差校正** | C#2 | PTQD, D2-DPM, TaC-QDM, QNCD | 建模量化噪声分布（线性/高斯），在采样公式中补偿 |
| **文本-图像一致性保持** | C#5 | DGQ | 保护 attention outlier + 对 `<start>` token 用对数量化 |

**关键方法串联逻辑**：PTQ4DM（开创性工作，提出校准策略）→ Q-Diffusion（解决 UNet concat 双峰）→ TDQ（时间动态量化参数，影响后续所有工作）→ TFMQ-DM（时间信息对齐）→ PTQD/D2-DPM（量化误差校正，利用去噪链结构补偿）。

### UNet-Based QAT（3 大方向）

| 方向 | 代表方法 | 核心思路 |
|------|---------|---------|
| **整体 QAT 优化** | Q-DM, QuEST, TuneQDM | 量化蒸馏 + 全局任务 loss；QuEST 证明 Taylor 展开在低比特下失效，需要参数微调 |
| **极低比特 DM** | BinaryDM, BiDM, BitsFusion | 二值化权重/潜空间；EBB + LRM 缓解表示坍塌 |
| **LoRA 增强** | EfficientDM, IntLoRA | 用 LoRA 降低 QAT 训练开销；IntLoRA 直接训练整数 LoRA |

**重要发现**：QuEST 证明了 BRECQ 的 Taylor 展开在低比特（W4A4）下失效——权重扰动过大导致条件不成立。这是 PTQ 在 W4A4 下普遍失败的理论解释。

### DiT-Based PTQ（2 大技术路线）

| 技术路线 | 代表方法 | 核心思路 |
|---------|---------|---------|
| **Group-wise 量化** | A-QDiT, Q-DiT | 调整量化粒度（group size）平衡 outlier 隔离与计算效率 |
| **Channel Equalization** | PTQ4DiT, DiTAS, [[ViDiT-Q]], HQ-DiT, SVDQuant | Channel 重平衡（smoothing/rotation/SVD）消除通道不均衡 |

本文将 [[ViDiT-Q]] 归入 Channel Equalization 路线，与 wiki 中的技术路线分类一致。

### DiT-Based QAT

**尚未有研究**。本文指出这是因为 DiT 参数量大，QAT 的计算和资源开销过高。但通过高效内存压缩技术探索 QAT 是未来有前景的方向。

## 统一基准实验

本文最大的差异化贡献之一是**统一基准评测**——在相同设置下对比所有开源方法，这在之前的扩散模型量化文献中从未系统做过。

### Class-Conditional（ImageNet 256×256，LDM-4）

| 方法 | Bits | IS↑ | FID↓ | sFID↓ | 要点 |
|------|------|-----|------|-------|------|
| FP | 32/32 | 366.03 | 11.13 | 7.83 | 基线 |
| D2-DPM | 8/8 | 333.89 | 8.12 | 7.92 | PTQ 最佳（误差校正类） |
| QuEST† | 8/8 | 365.70 | 10.93 | 7.95 | QAT 最佳，接近 FP |
| D2-DPM | 4/8 | 358.14 | 9.75 | 6.60 | PTQ W4A8 最佳 |
| QuEST† | 4/8 | 342.64 | 8.52 | 6.74 | QAT W4A8 最佳 |
| QuEST† | 4/4 | 197.02 | 7.76 | 9.89 | IS 大幅下降，PTQ 完全失败 |

### 关键发现

1. **W8A8 / W4A8 下 PTQ 与 QAT 差距不大**——PTQ 误差校正方法（D2-DPM）甚至在部分指标上优于 QAT
2. **W4A4 下 PTQ 完全失效**——只有 QAT（QuEST、EfficientDM）能工作，且质量显著下降
3. **量化噪声有时有益**——D2-DPM 在 W4A8 下的 IS 比 W8A8 高 24.25 分，sFID 降低 1.32。论文从 SDE 采样理论解释：量化噪声在 Langevin 扩散项中等价于额外随机性，可增强采样多样性
4. **LSUN 实验揭示方差爆炸**——使用 DDPM 采样（$\eta=1.0$）时，原生方差大，量化噪声更易导致误差爆炸；而 DDIM 采样（$\eta=0.0$）更稳定

### Text-Guided（SD v1-4，MS-COCO 512×512）

DGQ 在 W8A8 和 W4A8 下均为最佳，CLIP-score 最高——因为它专门保护了 cross-attention 的文本对齐能力。

## 量化伪影的定性分析

本文首次系统分类了扩散模型量化引入的**四类视觉伪影**：

| 伪影类型 | 原因 | 表现 |
|---------|------|------|
| **色彩偏移** | 量化噪声在 RGB 通道间不均匀（蓝色通道偏移更严重） | 整体色调偏暗黄 |
| **像素过曝** | 高方差激活跨通道叠加 + 极端值频率增加 | 对比度异常增强或噪点 |
| **结构特征改变** | Cross-attention 权重偏移影响空间域条件控制 | 生成对象形状/边界变化 |
| **细节模糊** | 连续激活映射到离散区域，丢失高频信息 | 纹理、毛发等细节丢失 |

**量化噪声的双面性**：PTQD 的 W4A8 结果展示了有趣的现象——适度方差增强了图像对比度（提升视觉质量），但过度方差产生噪点伪影。这与"量化噪声等价于 Langevin 扩散"的理论分析一致。

## 与 Wiki 已有内容的关系

### 与 [[Diffusion Model Quantization]] 概念页面

本综述与概念页面的已有内容高度互补：

- **概念页面**侧重 DiT 量化（2024-2025 年最新方法，ViDiT-Q/SVDQuant/Q-VDiT 等），且深入分析了技术路线对比
- **本综述**侧重 UNet 量化的完整方法谱系（PTQ4DM → D2-DPM 的演进链），且提供了统一基准实验和伪影分析

### 与 [[Low-bit Model Quantization Survey]]

| 维度 | Low-bit Survey | 本综述 |
|------|---------------|--------|
| 范围 | CNN/ViT/LLM/DM，DM 仅占 3.7 节 | 专注 DM，40 页深入分析 |
| 分类方式 | 按核心技术（8 大类） | 按架构×策略（UNet/DiT × PTQ/QAT） |
| 基准实验 | 无统一实验 | 3 任务统一评测，所有开源方法 |
| 独特贡献 | 覆盖面广，DM 方法列表全 | 6 挑战定义、伪影分类、轨迹分析 |

### 与 LLM 量化方法的技术交叉

本文大量引用 LLM 量化方法作为 DM 量化的技术来源：

- **BRECQ**：作为几乎所有 UNet PTQ 方法的 baseline（block-wise reconstruction）
- **[[SmoothQuant]]**：Channel smoothing 思想 → QNCD 的静态通道平滑、PTQ4DiT 的 CSB
- **[[AWQ]]**：激活感知 scaling → ViDiT-Q 的 channel balancing mask
- **[[QuaRot]]**：Hadamard 旋转 → ViDiT-Q 的 rotation-based balance
- **STE（Straight-Through Estimator）**：所有 QAT 方法的梯度近似基础
- **Taylor 展开失效**：QuEST 证明在低比特下 BRECQ 的理论保证不成立

## 未来方向

本文提出的五个方向：

1. **量化 + 高级训练策略（自监督/小样本学习）融合**——提升低资源场景下的鲁棒性
2. **QAT 微调增强**——结合微调保持量化后性能，平衡内存效率和生成质量
3. **向量量化在扩散模型中的应用**——对潜变量/嵌入做 VQ 可能带来更紧凑的表示（与 wiki 中 [[Lattice Codebooks]] 的思想相关）
4. **面向实际部署的硬件适配**——针对边缘设备和 AI 加速器的专用量化方案
5. **量化与可解释性的权衡**——理解量化如何影响模型决策的可解释性

## 引用关系

← 技术来源（LLM 量化）：[[SmoothQuant]]、[[AWQ]]、[[QuaRot]]（Channel Equalization 的 LLM 方法来源）
← 同领域综述：[[Low-bit Model Quantization Survey]]（3.7 节覆盖部分相同方法）
← 基准框架：BRECQ（几乎所有 UNet PTQ 方法的重建基线）
→ 代表方法页面：[[ViDiT-Q]]（DiT 量化代表）、[[Diffusion Model Quantization]]（概念页面）
→ 作者同期工作：D2-DPM（本文作者 Qian Zeng, Jie Song 的方法，量化误差双去噪校正）
→ 相关概念：[[Incoherence]]、[[Equivalent Scaling Transformation]]、[[Emergent Outlier Features]]
