---
title: 模型加速与压缩 Wiki 总览
aliases: [Overview, 总览]
tags: [model-compression, overview]
created: 2026-04-20
updated: 2026-04-29
sources: [LLM.int8(), GPTQ, SmoothQuant, AWQ, A Comprehensive Evaluation on Quantization Techniques for LLMs, SpinQuant, QuIP#, QuaRot, OSTQuant, FlatQuant, OmniQuant, Low-bit Model Quantization Survey, ViDiT-Q, Diffusion Model Quantization - A Review, VLM Quantization Best Practices, Q-VLM, SliM-LLM, KVQuant, KIVI, TurboQuant, H2O, SnapKV, ShadowKV, KVzip, TriAttention, StreamingLLM, Scissorhands, PyramidKV, GQA, SparseGPT, Wanda, DeepSeek-V2-MLA, SageAttention, SageAttention2, SageAttention3, QServe, Atom, Concentration-Alignment, AQLM, PTQ-Bench, KVTuner, KeepKV, KeyDiff, Expected Attention, TOVA, CAKE, DiffKV, EVICPRESS, AQUA-KV, DuoAttention, VL-Cache, Attention Debiasing, CAOTE, Ada-KV]
---

# 模型加速与压缩 Wiki 总览

本 wiki 是一个关于 **大语言模型加速与压缩（LLM Acceleration & Compression）** 的个人研究知识库。它追踪模型压缩、KV cache 管理、剪枝、架构效率等领域的方法、工具、理论与实践，通过持续录入来源文献逐步构建一幅完整的知识图谱。

## 领域概述

大模型推理面临四大瓶颈：

1. **模型权重内存**：70B 参数 FP16 需 140GB，超出单卡容量 → 量化、剪枝
2. **KV Cache 内存**：长上下文（128K-1M）下 KV cache 远超权重 → KV cache 量化、eviction、offloading
3. **计算吞吐量**：attention 和 FFN 的 FLOPs → 低比特计算、稀疏注意力、架构优化
4. **通信带宽**：多卡/多节点推理中的数据传输 → 压缩传输、KV cache 流式编码

本 wiki 覆盖的技术谱系：

```
模型加速与压缩
├── 一、量化（Quantization）           ← 降低比特数
│   ├── 权重量化（PTQ / QAT）
│   ├── 激活量化（W+A 联合量化）
│   ├── KV Cache 量化
│   ├── 扩散模型量化
│   └── 视觉-语言模型量化
├── 二、KV Cache 管理                  ← 压缩推理状态
│   ├── KV Cache 量化（同上）
│   ├── Token Eviction
│   ├── Low-Rank + Offloading
│   └── 架构级 KV 压缩（GQA/MLA）
├── 三、剪枝与稀疏化                   ← 移除冗余参数
│   ├── 非结构化剪枝
│   └── 结构化剪枝
└── 四、架构级效率改进                  ← 模型设计优化
```

---

## 一、量化（Quantization）

量化（Quantization）是将模型权重和/或激活值从高精度（如 FP32、FP16）映射到低精度表示（如 INT8、INT4、甚至更低）的技术，目的是在尽量保持模型质量的前提下，显著降低内存占用和推理开销。

随着大模型参数规模的膨胀（7B → 70B → 400B+），量化已成为模型部署的核心环节。主要分为两大范式：

- **[[Post-Training Quantization]] (PTQ)**: 训练完成后对权重进行量化，无需重新训练。代表方法包括 GPTQ、AWQ、SmoothQuant 等。
- **Quantization-Aware Training (QAT)**: 在训练过程中模拟量化效果，使模型学会适应低精度。

### 统一框架：两步分解

[[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 提出了一个极有价值的统一视角——将所有 PTQ 方法拆解为两步：

```
Step 1: 预量化变换                    Step 2: 量化误差补偿
┌─────────────────────────┐          ┌─────────────────────────┐
│  Shifting (平移)         │          │  RTN (直接四舍五入)       │
│  Scaling  (缩放)         │   ──→   │  GPTQ (Hessian 自补偿)   │
│  Rotation (旋转)         │          │  Low-rank (低秩补偿)      │
└─────────────────────────┘          └─────────────────────────┘
    使数据分布更适合量化                    补偿量化引入的误差
```

这让我们能用统一的分类方式理解所有方法。例如：SmoothQuant = Scaling + RTN；AWQ = Scaling + RTN；GPTQ = 无预处理 + GPTQ；OSTQuant = Scaling + Rotation + GPTQ。

### 方法演进

从首次发现 outlier 到系统化理论框架，该领域经历了三代演进：

**第一代：发现与应对 outlier（2022-2023）** —— [[LLM.int8()]]（NeurIPS 2022）首次发现 [[Emergent Outlier Features]] 并提出混合精度方案；[[GPTQ]]（ICLR 2023）用 Hessian 二阶信息实现 4-bit 权重压缩，使 175B 模型首次在单 GPU 上运行；[[SmoothQuant]]（ICML 2023）提出等价缩放变换来平滑激活 outlier，平滑变换可零开销融入 LayerNorm 实现 W8A8。 ![[images/Pasted image 20260422141211.png]]
**第二代：等价变换 + 感知量化（2023-2024）** —— [[AWQ]]（MLSys 2024 Best Paper）以激活感知缩放保护关键权重，calibration 鲁棒性最强；[[OmniQuant]]（ICLR 2024）首次将缩放因子升级为梯度优化的可学习参数；[[QuIP#]]（ICML 2024）引入 RHT + E₈ 格码本，将 weight-only 推到 2-3 bit；[[AQLM]]（ICML 2024）开创 learned codebook 向量量化路线，2-bit 全面超越 QuIP#；[[QuaRot]]（NeurIPS 2024）用计算不变性 + 随机 [[Hadamard Transform|Hadamard]] 实现零数据端到端 W4A4KV4。

**第三代：可学习变换 + 系统化理论（2024-2025）** —— [[SpinQuant]]（ICLR 2025）在 Stiefel 流形上学习最优旋转，W4A4KV4 仅比全精度差 2.9 分；[[OSTQuant]]（2025）从 QSUR 理论出发联合优化旋转与缩放；[[FlatQuant]]（ICML 2025）用 Kronecker 可学习仿射变换实现 LLaMA-3-70B W4A4 ≤1% drop。[[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 提出统一的两步分解框架，将所有方法纳入同一分类体系。

详细对比见 [[四篇经典 PTQ 方法对比]]。

### 方法体系：两步分解

所有 PTQ 方法都可拆解为 **预量化变换（Step 1）** 和 **量化误差补偿（Step 2）** 两步（见上方"统一框架"一节的示意图）。这是理解整个领域的核心骨架。

#### Step 1：预量化变换——2×2 分类矩阵

这是方法分化的核心维度。按**变换类型**（Rotation / Scaling）和**参数确定方式**（Unoptimized / Optimized）形成 2×2 分类：

```
              │  Unoptimized (公式/搜索)  │  Optimized (梯度学习)
─────────────┼──────────────────────────┼──────────────────────────────
  Rotation   │  QuIP#, QuaRot           │  SpinQuant, OSTQuant, FlatQuant
  (旋转)     │  随机 Hadamard / RHT     │  Cayley SGD / Riemann Adam
─────────────┼──────────────────────────┼──────────────────────────────
  Scaling    │  SmoothQuant, AWQ        │  OSTQuant, FlatQuant, OmniQuant
  (缩放)     │  α 公式 / grid search    │  learnable diag(s), 梯度优化
```

- **Rotation** 消除 outlier 方向，使通道分布趋向均匀，是 W4A4 的关键推动力 → 详见 [[Unoptimized Rotation]]、[[Optimized Rotation]]、[[Rotation-based Quantization]]
- **Scaling** 消除特征值幅度不均，平衡权重-激活量化难度 → 详见 [[Unoptimized Scaling]]、[[Optimized Scaling]]、[[Equivalent Scaling Transformation]]
- **最优组合**：Optimized Rotation + Optimized Scaling（[[Quantization Insights|Insight 1]]），但需注意优化一致性——混合使用未优化和优化组件可能适得其反（[[Quantization Insights|Insight 2]]）

#### Step 2：量化误差补偿

| 方法 | 原理 | 代表 | 推理开销 |
|------|------|------|---------|
| RTN | 直接四舍五入 | 所有方法的默认 baseline | 零 |
| GPTQ | Hessian 信息指导逐列补偿 | [[GPTQ]]，各方法标准 Step 2 | 零（改在权重上） |
| Low-rank | SVD 分解捕捉残差误差方向 | ZeroQuant-v2, LQER, CALDERA | ~3% FLOPs |

GPTQ 显著优于 Low-rank（PPL 11.80 vs 14.58），但二者贡献互补，叠加后可达 11.73（[[Quantization Insights|Insight 3]]）。

#### 正交维度：Mixed Precision

两步分解框架之外，还有一个重要的正交维度——**混合精度分配**。[[SliM-LLM]]（ICML 2025）发现 LLM 中 salient weights 呈空间聚簇分布，据此提出 group-wise 混合精度：按 group salience 排序分配 {N-1, N, N+1}-bit，以 KL 散度为优化目标。该策略可作为元方法叠加到 GPTQ（SliM-LLM）或 OmniQuant（SliM-LLM+）上，2-bit LLaMA-7B 从 152.31 PPL 降至 14.58。

### 核心概念网络

上述框架涉及的核心概念形成以下关联网络：

- **量化难点**：[[Emergent Outlier Features]]——~6.7B 参数后出现相变，驱动了所有预量化变换的研究
- **变换原语**：[[Hadamard Transform]]（旋转家族的共享计算核心，O(n log n) 复杂度）| [[Equivalent Scaling Transformation]]（缩放家族的共享数学结构）
- **理论基础**：[[Incoherence]]——量化的本质是消除矩阵中的"尖峰"使能量均匀分布，RHT 的 incoherence bound O(√log(mn)) 是目前理论最优 | [[Concentration-Alignment]]——SQNR 分解视角揭示量化误差由 Concentration（即 incoherence）和 Alignment（权重-激活方向交互）两个独立因素决定，现有旋转/缩放仅优化前者
- **量化格式**：[[FP4 Quantization]]（MXFP4/NVFP4, Blackwell 原生支持） | [[Lattice Codebooks]]（E₈ 格码本 VQ）——数据表示的前沿，且与变换类型存在深度交互（[[Quantization Insights|Insight 6-7]]）
- **量化粒度**：[[Quantization Granularity]]（per-tensor → per-group 的连续权衡）——与格式和变换选择共同决定最终精度
- **量化范式**：[[Post-Training Quantization]]（两步分解框架） | [[Weight-Only Quantization]]（GPTQ/AWQ 路线）

### 扩散模型量化

量化的应用范围已从 LLM 扩展到 **扩散模型**（Diffusion Models）。扩散模型的多步去噪过程带来三个独特挑战：时间步依赖的激活分布、误差沿去噪链累积、UNet/DiT 的架构异质性。

代表方法包括：PTQ4DM（CVPR 2023，首个扩散模型 PTQ）、Q-Diffusion（ICCV 2023）、SVDQuant（ICLR 2025，W4A4 + Nunchaku 引擎 3.5× 加速）、PTQ4DiT（NeurIPS 2024，首个 DiT 量化方法）、[[ViDiT-Q]]（ICLR 2025，首个 text-to-image/video DiT 量化，Static-Dynamic Channel Balance）。详见 [[Diffusion Model Quantization]]。

[[Low-bit Model Quantization Survey]] 的 3.7 节对该领域进行了系统分类，覆盖 20+ 篇论文。[[Diffusion Model Quantization - A Review]]（浙大，2025）则是首篇专门聚焦扩散模型量化的综合综述（40 页），提供了完整的 UNet/DiT 方法分类体系、统一基准实验和量化伪影定性分析。

### 视觉-语言模型量化

量化同样扩展到了**多模态大语言模型（MLLM / VLM）**。与纯 LLM 量化不同，VLM 量化面临 pipeline 级的新问题——ViT + Connector + LLM 各组件的敏感度不同，且组件间存在非加性交互效应。

[[VLM Quantization Best Practices]]（Maryland, 2026）在 BLIP-2 和 LLaVA 上系统研究了这一问题，核心发现：组件敏感度与参数量不成正比（ViT 仅 4.3% 参数但贡献 20-28% 重要性）；量化方法本身改变重要性分布（AWQ 将 80-98% 重要性集中到 LLM，GPTQ 更均衡）；任务特性决定最优比特分配（推理型任务偏向 LLM，对齐型任务更均衡）。GPTQ/AWQ 在 3.5-4.5 bpw 下仍能保持多模态性能。

[[Q-VLM]]（NeurIPS 2024，清华）则从方法角度提出了首个专门为 LVLM 设计的 PTQ 框架。其核心发现是 LVLM 中存在严重的**跨层依赖**——视觉编码器的量化误差沿 pipeline 非线性累积，使逐层独立量化失效。Q-VLM 用激活熵作为代理将层划分为强依赖 block 进行联合优化，在 LLaVA-7B W4A4 上达到 79.79%（ScienceQA），超越 QLoRA（77.53%）和 AWQ（74.02%），13B 模型实现 2.78× 内存压缩和 1.44× 加速。两篇工作互相验证——Best Practices 发现的"非加性交互效应"正是 Q-VLM 所建模的跨层依赖现象。

### 注意力计算量化

除了量化模型权重和 KV cache 的**存储**，量化还可以加速 attention 矩阵乘法的**计算**本身。SageAttention 系列（清华大学）是这一方向的代表，利用 GPU 低比特 Tensor Core 将 QK^T 和 PV 的矩阵乘法量化加速。详见 **[[Quantized Attention Kernel]]**。

三代演进：[[SageAttention]]（ICLR 2025, INT8 QK^T, ~2.1× FlashAttn2）→ [[SageAttention2]]（ICML 2025, INT4+FP8, ~3× FlashAttn2）→ [[SageAttention3]]（NeurIPS 2025, FP4 microscaling, **5× FlashAttn2**, 1038 TOPS on RTX5090）。核心技术是 smooth Q/K 消除 outlier 后用 INT 均匀量化，精度远优于 FlashAttention3 的 FP8（CosSim 99.55% vs 98.57%）。SageAttention3 还首次探索了 8-bit 可训练 attention（fine-tuning 无损，pretraining 收敛较慢）。

与 KV cache 量化、weight 量化、token eviction 完全正交，可叠加使用。

### 量化方法选型

**[[Quantization Method Selection]]** 从**问题出发**（而非从工具出发）系统整理如何根据面临的场景和约束选择量化方法。覆盖五大问题维度（精度目标、模型架构、部署约束、可用资源、精度恢复灵活性）、四类架构（Dense/MoE/VLM/Diffusion）的量化敏感性分布差异、从问题到方法的映射表、决策树、以及当前未解决的开放问题。与部署格式选型互补——前者关注"学术方法怎么选"，后者关注"工程格式怎么选"。

### 量化部署与选型

理论方法最终要落地为具体的部署格式。**[[Quantization Deployment Formats]]** 对比了 GPTQ、AWQ、GGUF（K-quants/I-quants）、EXL2、BitsAndBytes、FP8、AQLM、QuIP# 在推理硬件、引擎支持、量化成本和精度上的多维差异，并给出按场景（数据中心 / 单卡 / CPU / Mac / 微调 / 极限压缩）的选型决策框架。

**[[Model Quantization Ecosystem]]** 记录了三大主流开源 LLM 系列的量化生态现状：Qwen 系列（官方全格式覆盖，GPTQ+AWQ+GGUF+FP8）、DeepSeek 系列（原生 FP8 训练，MoE+MLA 架构带来独特挑战）、Gemma 系列（Google 官方 QAT INT4 checkpoint，训练时量化感知的范式创新）。也覆盖了 bartowski、unsloth 等社区量化者的工具与方法。

**[[QServe]]**（MLSys 2025）代表了"量化算法-系统协同设计"的趋势——QoQ W4A8KV4 渐进式量化 + 定制 CUDA kernel，在 L40S 上实现 1.5-3.5× 吞吐提升（vs TensorRT-LLM），证明了 serving 场景下去量化开销是比量化位宽更关键的瓶颈。**[[Atom]]**（MLSys 2024）则是该方向的先驱，通过 W4A4 + outlier 通道重排序 + 混合精度 + 定制 INT4 kernel，首次实现了大 batch serving 场景下最高 7.7× 的吞吐提升（vs FP16）。

**[[AQLM]]**（ICML 2024）开创了 LLM 向量量化的路线——将 Additive Quantization 从信息检索扩展到 LLM 压缩，通过 learned codebook + input-aware 优化 + intra-block fine-tuning，首次在 2-bit 区间实现 Pareto 最优（LLAMA2-70B 仅 +0.82 PPL vs FP16），证明了超越标量量化的向量量化范式在极限压缩下的巨大潜力。

### 关键量化发现与趋势

基于 [[Quantization Insights]] 中汇总的 9 条系统实验发现，可归纳为三个元原则：

**元原则一：分布匹配——量化的本质是让数据匹配网格。** 量化格式（INT/FP）、粒度（group size）、预量化变换（rotation/scaling）本质上都在追求数据分布与量化网格的最佳匹配。具体表现为：激活天然非对称，对称量化浪费一半范围，**权重对称 + 激活非对称**是最佳工程折中（Insight 4）；粒度是精度-存储的连续权衡，group-128 是常用平衡点（Insight 5）；FP4 在粗粒度下远优于 INT4（适配正态分布），但细粒度下差距消失（局部分布趋于均匀，Insight 6）；旋转对 FP4 几乎无效——小 group 已局部隔离 outlier，**旋转后 INT4 反而优于 FP4**，这意味着 FP4 时代需要全新预处理范式（Insight 7）。

**元原则二：互补组合——多技术叠加优于单一极致。** 不同技术解决不同维度的问题（outlier 方向 vs 幅度不均 vs 误差补偿），贡献累加。Optimized Rotation + Optimized Scaling 是最优预量化组合，QSUR 只有二者同时作用时才达到最大值（Insight 1）；GPTQ 优于 Low-rank，但叠加后进一步降低 PPL（Insight 3）；最优 W4A4 配置下 LLaMA-3.1-8B 的 PPL 差距仅 +1.0，且随模型增大通常递减（但 70B 反而 +1.6，可能因多层误差累积）（Insight 8）。

**元原则三：优化一致性——要么都不优化，要么都优化。** 组合中各组件需保持优化状态的一致性。未优化旋转后做 calibrated scaling 反而使 PPL 恶化（旋转使分布变均匀后 calibration 假设不再成立，Insight 2）；好消息是整个预处理链的推理开销极低——绝大部分可融入权重，运行时仅需少量在线 Hadamard 变换（~8%）和可选 Low-rank 分支（~3%），W4A4 在部署中性价比极高（Insight 9）。

**补充发现（[[PTQ-Bench]]）**：模型规模-比特宽度存在明确权衡——2-bit 最大模型始终不如 4-bit 最小模型（同系列），但 3-bit 仍保持 scaling law 优势。此外，PTQ 策略的适用性强依赖模型训练充分度：欠训练模型（LLaMA-1/2）适合 Rotation，充分训练模型（LLaMA-3/3.1）适合 Compensation。跨结构场景（MoE/Mamba）中，Compensation (GPTQ) 是唯一普适方案。

---

## 二、KV Cache 管理

随着长上下文需求的爆发（128K→1M→10M），KV cache 成为推理的核心内存瓶颈——LLaMA-7B 在 1M context 下 KV cache 需 128GB，远超模型权重（14GB）。详见 **[[KV Cache Management]]**。

KV cache 管理涵盖四大正交子方向，可自由组合实现乘法压缩：

```
总 KV cache ∝ (层数) × (KV head 数) × (token 数) × (每 entry 比特数)
               架构级 GQA/MLA    Token Eviction   KV Cache 量化
                                   ↕ Low-Rank/Offload (分层存储)
```

### KV Cache 量化

降低每个 KV entry 的比特数。核心发现：Key per-channel、Value per-token 量化（[[KVQuant]]、[[KIVI]] 独立验证）。[[TurboQuant]] 从信息论出发建立理论最优性保证。[[KVTuner]]（ICML 2025）揭示了 layer-wise 敏感度差异并通过离线 MOO 搜索实现 3.25-bit 近无损量化。[[DiffKV]]（OSDI 2025）将量化与 pruning 统一——利用 Key>Value 影响力差异（K8V4/K4V2）+ 层次化 token compression + per-head 动态稀疏性，配合 Parallel KV Compaction 系统实现 2.7×–5.7× 压缩和 5.4× 吞吐提升，在 thinking model 上近无损。[[AQUA-KV]]（2025）开辟层间预测新范式——利用相邻层 KV 的强线性依赖训练紧凑 predictor，只量化残差而非原始值，在 2-bit 下 Qwen2.5 7B LongBench 46.43（vs HIGGS 直接崩溃 25.97），可与任意后端量化器和 pruning 方法组合。详见 **[[KV Cache Quantization]]**。

### Token Eviction

基于 [[Attention Sparsity]]（>95% attention 权重接近零），移除不重要 token 的 KV。方法沿三个轴分类：query 依赖性（query-aware [[H2O]]/[[SnapKV]] vs query-agnostic [[KVzip]]/[[TriAttention]]）、eviction 时机（prefill vs generation）、budget 分配（fixed vs layer-adaptive [[PyramidKV]]）。[[StreamingLLM]] 发现的 [[Attention Sink]] 现象是该领域的基础性观察。[[TOVA]]（arXiv 2024）从 Multi-State RNN 理论出发，每步丢弃 attention score 最低的 token，无需手工设计 window 或 sink 规则，1/8 cache 即可近无损。[[KeepKV]]（AAAI 2026）将 eviction 升级为 merging——通过 Electoral Votes + ZIP-Merging 实现单步无损压缩，避免不可逆信息丢失，可与任意 eviction 策略组合使用。[[KeyDiff]]（NeurIPS 2025）提出 attention-free 的评分方式——Key 几何多样性作为重要性代理，在资源受限的 block prompt processing 场景下远优于 attention-based 方法。[[Expected Attention]]（arXiv 2025, NVIDIA）通过 hidden state 高斯分布假设以闭合形式预测未来 query 的期望注意力，同时适用于 prefill 和 decode 两阶段，在多模型上实现 SOTA 压缩-性能 trade-off。[[CAKE]]（ICLR 2025）首次系统建模 layer 间注意力异质性，基于空间分散度和时间偏移量自适应分配 cache budget（P2A 策略），并通过 cascading management 在 prefilling 中动态管理，仅 3.2% cache 即保持性能。[[DuoAttention]]（ICML 2025, MIT Han Lab）将 heads 二分类为 Retrieval Heads（需完整 KV）和 Streaming Heads（仅 sink+recent），通过 optimization-based 识别实现 head-level 差异化 eviction，MHA 2.55× / GQA 1.67× 内存压缩且长上下文无损。[[CAOTE]]（Qualcomm AI Research, arXiv 2025）首次以闭合形式将 attention score 和 value 向量整合为统一 eviction 评分——证明 eviction error 与 CAOTE score 精确相等（Theorem 3.2），作为 meta-heuristic 可叠加在任何 attention-based 方法上，H2O+CAOTE 在 LongBench 2K budget 下提升超 100%。[[Ada-KV]]（NeurIPS 2025, USTC）建立了 Top-k eviction 的 L₁ loss upper bound 理论框架，并据此提出首个 head-wise adaptive budget allocation 策略——跨 head 全局 Top-B 选择自适应分配 cache budget，plug-and-play 增强 SnapKV/PyramidKV，在 Ruler 和 LongBench 的 question-agnostic 场景下一致提升，被 CriticalKV/DefensiveKV/KVzip/Expected Attention 等后续工作广泛采纳。[[EVICPRESS]]（UChicago/Microsoft 2025）提出联合优化 compression 和 eviction 的系统——per-context 自适应选择压缩方法/比率和存储层级放置，通过 utility function 全局优化 quality-TTFT trade-off，在 12 个 LongBench 数据集上 TTFT 降低 1.43×–3.77×。[[VL-Cache]]（arXiv 2024, UCLA/Amazon）是首篇专门为 VLM 优化 KV cache compression 的工作——发现 VLM 中 modality boundary 现象（视觉 token 间均匀 attend、语言 token 对视觉 token 高度集中），提出 sparsity-aware layer budget allocation + accumulated post-vision attention scoring，10% cache 即保持 98%+ 精度，decoding 7.08× 加速。详见 **[[Token Eviction]]**。

### Low-Rank Compression + Offloading

[[ShadowKV]] 发现 Pre-RoPE Key 具有低秩性，将低秩部分存 GPU、残差 offload 到 CPU，配合 landmark-based sparse retrieval 实现 6× bandwidth 提升。详见 **[[Low-Rank KV Compression]]** 和 **[[KV Cache Offloading]]**。

### VLM KV Cache 管理

Vision-Language Model 的 KV Cache 管理与纯 LLM 存在根本性差异——VLM 的多模态 prompt 结构导致独特的 **modality boundary** 现象：视觉 token 间近乎均匀 attend，而语言→视觉的注意力高度集中于少数关键 patch。这意味着 LLM 的 eviction 方法（如 H2O、SnapKV）直接迁移到 VLM 会产生次优结果。[[VL-Cache]]（UCLA/Amazon, arXiv 2024）首次系统揭示此现象，提出 post-vision attention scoring + sparsity-aware layer-adaptive budget allocation，10% KV cache 即保持 98%+ 精度，7.08× decode 加速。[[MixKV]]（SJTU, ICLR 2026）发现 VLM 中 importance-only 方法导致语义覆盖丢失，提出 head-wise 冗余度自适应的 importance+diversity 联合评分，极端压缩(budget=64)下平均 +5.1%。[[AirCache]]（Alibaba, arXiv 2025）进一步提出 elite observation window（self-attention 筛选关键 text tokens）+ strength/skewness budget allocation，在 LLaVA-OV/InternVL2/Qwen2-VL 三种架构上验证有效，1% cache 下比 SnapKV 高 6.5%。[[Attention Debiasing]]（Shanghai U/Nankai, arXiv 2026）系统揭示 VLM attention 中的 recency bias 和 padding attention sink 两种系统性偏差，提出 plug-and-play 的指数去偏+padding 置零，一致提升 6 种 token pruning 方法。详见 **[[VLM KV Cache]]** 子目录。

### 架构级 KV 压缩

MQA → [[GQA]] → [[DeepSeek-V2-MLA|MLA]] 的演进从模型设计层面减少 KV cache，GQA 已成为现代 LLM 标准配置。详见 **[[Grouped-Query Attention]]**。

### 经验观测汇总

**[[KV Cache Empirical Observations]]** 系统整理了所有 KV cache 管理研究中发现的经验性观测事实，涵盖 Key/Value 统计特性、Pre-RoPE 结构化性质、attention 稀疏性与 sink 现象、层间/头间异质性、时间动态、context 级差异等 13 大类 40+ 条观测，并分析了观测间的因果和互补关系。这些观测是所有 KV cache 压缩方法的共同理论基础。

### Pre-RoPE 结构化特性

多篇独立工作汇聚于同一发现：Pre-RoPE Key 具有丰富的结构化特性（通道稳定性、低秩性、方向集中性），而 RoPE 的位置旋转会破坏这些结构。这成为量化（[[KVQuant]]）、低秩压缩（[[ShadowKV]]）和 eviction scoring（[[TriAttention]]）的共同理论基础。

---

## 三、剪枝与稀疏化

剪枝（Pruning）通过移除冗余权重或结构来压缩模型，与量化正交互补。详见 **[[Pruning and Sparsity]]**。

目前录入两篇代表性工作：

- **[[SparseGPT]]**（ICML 2023）：基于 Hessian 信息的 one-shot 非结构化剪枝，50% 稀疏度下近无损。其 Hessian 逆更新框架与 [[GPTQ]] 共享数学结构
- **[[Wanda]]**（ICLR 2024）：Weight × Activation 幅度剪枝，零 calibration，与量化中 [[AWQ]] 的 salient weight 识别思想一脉相承

**与量化的关系**：剪枝减少非零参数数量，量化降低每个参数的比特数，两者可组合——如 50% 稀疏 + 4-bit 量化 = 理论 16× 压缩。SparseGPT 和 Wanda 均已与 GPTQ 组合验证有效性。

---

## 四、架构级效率改进

从模型架构设计层面提升推理效率，不依赖后处理压缩。本方向目前作为 stub，将随后续论文录入扩展。

已录入的架构级工作主要集中在 KV cache 方向：[[GQA]]（Grouped-Query Attention）、[[DeepSeek-V2-MLA]]（Multi-head Latent Attention）——详见 [[Grouped-Query Attention]]。

---

## 工具与框架

- [[bitsandbytes]]：LLM.int8() 参考实现，HuggingFace 生态量化基础设施
- [[ExLlamaV2]]：高性能本地推理引擎 + EXL2 混合精度量化格式，消费级 GPU 上速度最快的方案之一
- TinyChat：AWQ 配套 4-bit 推理框架，桌面/边缘 GPU 3.5-3.7× 加速
- Nunchaku：SVDQuant 配套 W4A4 推理引擎，FLUX.1-12B 在 4090 上 3.5× 加速
- 工业集成：SmoothQuant 已进入 NVIDIA TensorRT-LLM、Microsoft ONNX Runtime 等主流框架

## 综述互补

本 wiki 同时录入了多篇综述，它们提供互补的视角：

- **[[A Comprehensive Evaluation on Quantization Techniques for LLMs]]**：聚焦 LLM PTQ，提出两步分解统一框架（Step 1: 预量化变换 + Step 2: 误差补偿），是本 wiki 量化知识体系的核心骨架
- **[[PTQ-Bench]]**：从计算策略角度将 weight-only PTQ 分为四大类（Compensation/Rotation/Salience/Optimization），通过跨比特宽度、跨结构、跨模态三维评估揭示各策略的鲁棒性特征。核心发现：Rotation + Compensation 组合是统一鲁棒最优方案；2-bit 大模型不如 4-bit 小模型
- **[[Low-bit Model Quantization Survey]]**：覆盖 CNN/ViT/LLM/Diffusion，按核心技术分为 8 大类 24 子类，视野更广但不做统一框架抽象
- **[[Diffusion Model Quantization - A Review]]**：首篇专门聚焦扩散模型量化的综合综述（浙大，40 页），按架构×策略（UNet/DiT × PTQ/QAT）分类，含统一基准实验和量化伪影分析

综述间的互补关系：两步分解框架回答"方法在数学上做了什么"（变换类型+补偿方式），PTQ-Bench 回答"什么场景该选什么策略"（工程鲁棒性），Low-bit Survey 提供全景分类，Diffusion 综述聚焦特定领域。PTQ-Bench 的实验结论（Rotation+Compensation 最优）直接验证了两步分解框架的核心 insight（Step 1 变换 + Step 2 补偿正交互补）。

## 当前状态

- 已录入来源数：**56**
- Wiki 页面数：**93**
- 覆盖领域：量化、KV Cache 管理、注意力计算加速、剪枝与稀疏化、架构效率、量化部署与选型
- 上次更新：2026-05-15

## 目录结构

本 wiki 按**主题**组织，量化目录下设子分类：

```
wiki/
├── overview.md
├── entities/                     # 跨领域研究机构
│   ├── MIT Han Lab.md
│   └── Cornell RelaxML Lab.md
├── 量化/
│   ├── 理论与框架/               # 17 页：核心概念、预量化变换理论、方法选型
│   ├── 方法/                    # 14 页：具体量化方法论文
│   ├── 多模态与扩散/             # 5 页：VLM、扩散模型量化
│   ├── 注意力计算量化/           # 4 页：SageAttention 系列
│   ├── 部署与生态/               # 5 页：部署格式、模型生态、GGUF、ExLlamaV2、工具
│   └── 综述与对比/               # 4 页：综述论文、方法对比
├── KV Cache 管理/                # 36 页
│   └── VLM KV Cache/            # 5 页：VLM 专用 KV Cache 管理（modality-aware）
├── 剪枝与稀疏化/                 # 3 页
└── 架构级效率改进/               # 3 页
```
