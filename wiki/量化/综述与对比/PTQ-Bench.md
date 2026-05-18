---
title: "PTQ-Bench: Benchmarking Post-Training Quantization in LLMs"
aliases: [PTQ-Bench, Benchmarking PTQ in LLMs]
source_type: paper
source_url: "https://arxiv.org/abs/2502.13178"
source_date: 2025-05-21
ingested: 2026-05-11
venue: arXiv preprint (under review)
authors: [Jiaqi Zhao, Ming Wang, Miao Zhang, Yuzhang Shang, Xuebo Liu, Yaowei Wang, Min Zhang, Liqiang Nie]
affiliation: Harbin Institute of Technology (Shenzhen) / Illinois Institute of Technology
code: "https://github.com/zjq0455/PTQ-Bench"
tags: [model-compression, quantization, survey, benchmark, post-training, taxonomy, evaluation, weight-only]
---

## 核心要点

PTQ-Bench 是一个针对 LLM **权重量化（weight-only PTQ）** 方法的综合基准，核心贡献：

1. **四策略分类法**：按计算策略将 PTQ 方法分为 Compensation-based、Rotation-based、Salience-based、Optimization-based 四大类
2. **三维鲁棒性评估**：跨比特宽度（cross-bitwidth）、跨结构（cross-structure）、跨模态（cross-modality）的统一基准
3. **关键发现**：模型规模-比特宽度权衡（2-bit 大模型不如 4-bit 小模型）；Compensation + 其他策略的组合是统一鲁棒的最优方案

覆盖模型：LLaMA-1/2/3/3.1（7B-70B）、Mixtral-8×7B、DeepSeekMoE-16B、Mamba-1.4B/2.8B、LLaVA-1.5、VILA-1.5。

## 四策略分类法（Taxonomy）

与 [[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 的"两步分解"框架互补，PTQ-Bench 从**计算策略/优化机制**角度提出分类：

### 1. Compensation-based（误差补偿）

代表方法：**[[GPTQ]]**、VPTQ、APTQ、QuantEase

核心策略：在量化过程中动态更新未量化权重以补偿量化误差。GPTQ 利用 Hessian 信息计算最优补偿量：

```
δ = -[w_q - quant(w_q)] / [H⁻¹]_qq · (H⁻¹)_{:,q}
```

逐列量化权重，同步补偿当前 block 内的残余权重。

### 2. Rotation-based（旋转变换）

代表方法：**QuIP**、[[QuIP#]]、[[QuaRot]]、[[SpinQuant]]、DuQuant

核心策略：源于 [[Incoherence]] 观察——权重矩阵具有 µ-incoherence 性质时量化效果更好。通过正交变换（Hadamard/Kronecker 旋转）消除 outlier，使权重在各方向幅度均匀。

```
µ-incoherence: max|W_ij| ≤ µ·‖W‖_F / √(mn)
```

### 3. Salience-based（显著性感知）

代表方法：**[[AWQ]]**、OWQ、SpQR、SqueezeLLM、PB-LLM、[[SliM-LLM]]

核心策略：识别权重中的"重要"参数并差异化处理。AWQ 用输入激活幅度作为 saliency 指标，通过等价缩放保护 salient weights：

```
Q(w·s) · x/s = Δ · Round(ws/Δ) · x · 1/s
```

避免了混合精度部署的硬件不友好问题。

### 4. Optimization-based（优化框架）

代表方法：**OmniQuant**、CBQ、LRQuant、AffineQuant

核心策略：引入可学习参数（如 clipping range、scaling factors），通过 block-wise 梯度优化更新量化参数：

```
argmin_{Θ₁,Θ₂} ‖F(W,X) - F(Q_w(W;Θ),X)‖
```

以全精度 block 输出为监督信号，高效训练。

### 分类法与两步分解框架的映射

| PTQ-Bench 分类 | 两步分解框架中的对应位置 |
|---------------|----------------------|
| Compensation-based (GPTQ) | Step 2: 量化误差补偿 |
| Rotation-based (QuIP) | Step 1: 预量化变换（Rotation） |
| Salience-based (AWQ) | Step 1: 预量化变换（Scaling） |
| Optimization-based (OmniQuant) | Step 1: 可学习变换（Optimized Scaling） |

两种分类体系互补：两步分解关注"做了什么操作"（变换类型+补偿方式），PTQ-Bench 关注"用什么策略优化"（计算机制），后者更侧重方法的工程鲁棒性特征。

## 三维鲁棒性评估结果

### Cross-Bitwidth Robustness（跨比特宽度鲁棒性）

**4-bit**：所有策略表现接近，AWQ 略优。

**3-bit**：策略开始分化。AWQ 仍保持优势，OmniQuant 在 LLaMA-3/3.1 上崩溃（3-bit LLaMA3-70B: OmniQuant 7.4e4 PPL vs GPTQ 7.16 PPL）。

**2-bit（核心分化区间）**：
- **Salience-based (AWQ) 全面崩溃**：所有模型上完全丧失语言能力（PPL ~10⁵）
- **Optimization-based (OmniQuant) 在 LLaMA-3/3.1 上崩溃**：虽然在 LLaMA-1/2 上尚可，但在更充分训练的 LLaMA-3/3.1 上完全失效
- **Rotation-based (QuIP) 和 Compensation-based (GPTQ) 保持可用**：表现因模型训练充分度而异

**关键发现：训练充分度决定低比特鲁棒性策略选择**

| 模型训练状态 | 2-bit 最优策略 | 原因 |
|------------|-------------|------|
| 欠训练 (LLaMA-1/2) | **Rotation-based (QuIP)** | QuIP 7.70 vs GPTQ 9.00 (LLaMA2-70B) |
| 充分训练 (LLaMA-3/3.1) | **Compensation-based (GPTQ)** | GPTQ 23.43 vs QuIP 53.18 (LLaMA3-70B) |

论文引用 quantization scaling law 解释：LLaMA-3/3.1 训练数据量数倍于 LLaMA-1/2，更充分的训练导致信息更密集，2-bit 下信息损失更大。

### Cross-Structure Robustness（跨结构鲁棒性）

| 结构 | AWQ | GPTQ | QuIP | OmniQuant |
|------|-----|------|------|-----------|
| Mamba | ❌ 不支持 | ✅ 稳定 | ✅ 稳定 | ❌ 崩溃 |
| MoE (Mixtral) | ❌ 不支持 | ✅ **最优** | ⚠️ 不稳定 | ⚠️ 高比特可用 |
| MoE (DeepSeek) | ❌ 不支持 | ✅ 稳定 | ✅ 2-bit 优势 | ⚠️ 不稳定 |

AWQ 失败原因：
- **Mamba**：in_proj 前只有 RMSNorm（无线性层），无法融入 scaling 参数保证输出不变性
- **MoE**：router 动态选择 expert，无法离线融合 scaling 超参

OmniQuant 在 Mamba 上完全崩溃（PPL ~10⁴），在 MoE 上 2-bit 也严重退化。

**结论**：Compensation-based (GPTQ) 是跨结构鲁棒性最强的策略；Rotation-based (QuIP) 在 2-bit 下有优势但在 MoE（特别是 Mixtral）上不稳定。

### Cross-Modality Robustness（跨模态鲁棒性）

在 VILA-1.5 和 LLaVA-1.5 上的 MLLM 量化结果：

| 比特宽度 | 趋势 |
|---------|------|
| 3-4 bit | 所有策略表现接近，差异可忽略（最大 gap 1.7% on LLaVA1.5-13B） |
| 2-bit | 仅 GPTQ 和 QuIP 保持有效推理能力；AWQ 和 OmniQuant 完全崩溃 |

示例：VILA1.5-13B 2-bit：AWQ 7.69% vs QuIP **54.17%** vs GPTQ 45.93%

## 关键发现：Beyond Benchmarking

### 发现 1: Model Size-Bitwidth Tradeoff

**2-bit 大模型始终不如同系列 4-bit 小模型**，无论使用何种 PTQ 策略：

- LLaMA-65B @ 2-bit (OmniQuant): 9.37 PPL / 42.81% Acc
- LLaMA-7B @ 4-bit (OmniQuant): 6.61 PPL / 51.01% Acc

这一规律在所有模型系列、所有结构、所有 PTQ 策略下普遍成立。

**但 3-bit 仍然保持 scaling law 优势**：
- LLaMA2-13B @ 3-bit (GPTQ): 6.28 PPL / 55.44% → 优于 LLaMA2-7B @ 4-bit: 6.45 PPL / 52.86%

**实践含义**：
1. 当前 2-bit PTQ 对超大模型的研究价值有限（部署不如用 4-bit 小模型）
2. 3-bit 是有效的压缩目标，仍能享受大模型规模优势
3. 极低比特 PTQ 需要专门为大模型设计的方法突破

### 发现 2: Compensation 是统一鲁棒的基础策略

PTQ-Bench 进一步测试了策略组合：

| 策略 | LLaMA3.1-8B W2 | Mixtral-8×7B W2 |
|------|----------------|-----------------|
| GPTQ (单独) | 510.33 / 23.20% | 22.92 / 28.19% |
| OWQ (Salience) | 305.77 / 24.33% | 35.60 / 32.89% |
| LRQuant (Optim.) | 193.83 / 26.05% | 41.81 / 29.77% |
| PB-LLM (Compen.+Salien.) | 84.45 / 33.27% | — |
| **QuaRot+GPTQ** (Compen.+Rot.) | **33.33 / 35.28%** | — |
| **QuIP+GPTQ** (Compen.+Rot.) | — | **9.59 / 45.26%** |

**结论**：Compensation-based 作为"基础层"与 Rotation-based 组合可在各场景下获得 SOTA 跨比特、跨结构鲁棒性。这验证了 [[A Comprehensive Evaluation on Quantization Techniques for LLMs]] 中"Step 1 变换 + Step 2 补偿互补"的理论框架。

## 策略选型推荐总结

| 场景 | 推荐策略 | 备注 |
|------|---------|------|
| 4-bit 通用部署 | **AWQ (Salience)** | 简单高效，精度略优 |
| 3-bit 通用部署 | **AWQ** 或 **GPTQ** | AWQ 仍有优势 |
| 2-bit 欠训练模型 (LLaMA-1/2) | **QuIP (Rotation)** | 低比特鲁棒性最强 |
| 2-bit 充分训练模型 (LLaMA-3+) | **GPTQ (Compensation)** | 信息密集模型更需补偿 |
| 非标准结构 (Mamba/MoE) | **GPTQ (Compensation)** | 唯一全结构兼容 |
| 多模态模型 2-bit | **QuIP** 或 **GPTQ** | 其他策略崩溃 |
| 追求最优鲁棒性 | **Rotation + Compensation 组合** | QuaRot+GPTQ 或 QuIP+GPTQ |

## 与本 wiki 其他工作的关系

**互补视角**：
- [[A Comprehensive Evaluation on Quantization Techniques for LLMs]]（两步分解框架）关注"变换+补偿的组合空间"——回答"方法在数学上做了什么"
- **PTQ-Bench**（四策略分类）关注"策略的工程鲁棒性特征"——回答"什么场景该选什么策略"

**验证一致性**：
- PTQ-Bench 的"Compensation + Rotation 最优组合"结论完美验证了两步分解框架的核心 insight：Step 1（变换）+ Step 2（补偿）正交互补
- PTQ-Bench 的"OmniQuant 在新模型上崩溃"也与两步分解框架 Insight 2（优化一致性）呼应

**与其他页面的关联**：
- [[Rotation-based Quantization]]——PTQ-Bench 验证了旋转方法的低比特鲁棒性优势
- [[GPTQ]]——被证明是跨结构鲁棒性最强的基础方法
- [[Quantization Method Selection]]——PTQ-Bench 提供了按场景选择策略的实验依据
- [[Concentration-Alignment]]——理论上解释了为什么 Rotation 在低比特更有效（提升 Concentration）
