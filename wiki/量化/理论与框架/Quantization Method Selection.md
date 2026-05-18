---
title: 量化方法选型指南
aliases: [Quantization Method Selection, 量化方法选择, 如何选择量化方法]
tags: [model-compression, quantization, concept, guide, decision-framework]
created: 2026-05-08
updated: 2026-05-08
sources: [A Comprehensive Evaluation on Quantization Techniques for LLMs, SmoothQuant, AWQ, GPTQ, QuaRot, SpinQuant, OSTQuant, FlatQuant, QuIP#, QServe, OmniQuant, KVQuant, KIVI]
---

## 概述

本页面从**问题出发**而非从工具出发，系统整理如何根据面临的具体场景和约束选择量化方法。与 [[Quantization Deployment Formats]]（关注成熟框架的工程选型）互补，本页面聚焦学术前沿方法的适用条件和局限性，帮助在研究中做出合理的方法决策。

核心原则：**量化方法的选择不应该从"哪个框架流行"出发，而是从你面对的问题维度出发。**

---

## 一、驱动方法选择的问题维度

### 1. 精度目标（Bit-width Target）

量化精度不是连续的——在不同 bit-width 区间，瓶颈和最优方法截然不同：

| Bit-width 区间 | 核心瓶颈 | 最优方法族 |
|---------------|---------|-----------|
| W8A8 | Activation outlier | [[SmoothQuant]] / FP8 |
| W4A16 | Weight distribution | [[GPTQ]] / [[AWQ]] |
| W4A8 / W4A4 | Activation + Weight 联合 | [[Rotation-based Quantization]] + GPTQ |
| W2-3 | 信息论极限 | Vector Quantization（[[QuIP#]] / AQLM） |

关键断点：4-bit 是 scalar quantization 的"甜区"；3-bit 以下 scalar 方法开始崩溃，必须转向 VQ/codebook 方法。

### 2. 模型架构特征

不同架构具有截然不同的量化敏感性分布（详见下方第二节）。架构决定了：哪些组件是量化瓶颈、是否需要 mixed-precision、校准策略是否充分。

### 3. 部署约束

- **推理硬件**：GPU（INT8/FP8 Tensor Core）vs CPU（AVX2/NEON）vs NPU → 决定能用的量化格式
- **Memory budget**：决定能容忍的 bit-width 下限
- **Latency budget**：决定是否可接受在线变换（如 Hadamard）或 dequantization 开销
- **Batch size**：小 batch = memory-bound（weight-only 有效）；大 batch = compute-bound（需要 W+A 量化）

### 4. 可用资源

| 资源条件 | 推荐路线 |
|---------|---------|
| 无校准数据 | RTN / TurboQuant（data-oblivious） |
| 少量校准数据（16-128条） | [[AWQ]] / [[GPTQ]] / [[QuaRot]] |
| 可承担几小时 GPU 计算 | [[SpinQuant]] / [[OSTQuant]] / [[FlatQuant]]（优化旋转+缩放） |
| 有预训练 1-5% 的计算预算 | QAT（LLM-QAT / BitDistiller / EfficientQAT） |

### 5. 精度恢复灵活性

是否允许引入额外推理开销来换取精度？
- 零开销路线：Scaling（融入权重）+ GPTQ（修改权重）
- 微开销路线：在线 Hadamard（~8% latency）
- 有开销路线：Low-rank 残差分支（~3% FLOPs）

---

## 二、不同架构的量化敏感性分布

### Dense Transformer（标准 LLM）

#### 组件级敏感性

**Embedding / lm_head 层——最难量化的组件：**

这两层难量化的本质原因并非简单的"weight magnitude 大"，而是三个结构性因素：

1. **误差传播位置**：Embedding 是整个模型的输入门，其量化误差会向后传播并在所有后续层中累积放大；lm_head 是输出门，直接决定 token 概率分布，微小误差改变 logit 排序
2. **Per-tensor 粒度灾难**：vocab_size 极大（100K-150K），embedding 矩阵 (vocab_size × hidden_dim) 的不同 row 之间 scale 差异巨大。Per-tensor 量化时，单一 scale 无法同时适配所有 token 的动态范围，必须用 per-row 量化
3. **低频 token 的分布异常**：高频词（如 "the"）的 embedding 经过充分训练后 norm 稳定；低频词/罕见词的 embedding 可能仍停留在随机初始化附近，产生大量分布异常点

实践中 GPTQ/AWQ 通常对 embedding 层做特殊处理或直接跳过量化。

**Q/K 投影 activation——比 V/O 更难量化：**

Q/K 难量化的原因**不是 RoPE**（RoPE 是正交变换，‖Rx‖ = ‖x‖，不改变幅度，不引入额外 outlier）。真正原因是 Q·K^T 的功能需求：

1. **Attention score 的极端分布需求**：softmax 需要输出近似 one-hot 才能实现精确的位置选择。这要求 Q·K^T 的某些元素远大于其他元素（差值需 > 5-10）。量化噪声如果改变了最大值的排序，attention 就会看错位置——这是一个**放大器**而非平均器
2. **Massive Activation 现象**（Sun et al., 2024）：模型学会用特定维度做"全局 attention dump"（所有 query 都给 sink token 打高分），这些维度的 Q/K 值系统性地大，但只集中在少数维度
3. **Head 间的异质性**：不同 attention head 承担不同功能（local attention、global retrieval、syntactic pattern），其 Q/K 分布形态差异极大，per-tensor 量化无法覆盖所有 head 的动态范围

**V/O 投影 activation——相对易量化：**

V 和 O 参与的是**加权求和**而非**比较/选择**操作。V 矩阵的值被 attention weight 做线性组合，这个操作对 V 中的小量化噪声有天然的"平均化"抑制效果（类似 ensemble smoothing）。相比之下 Q·K^T 做的是 argmax-like 的选择操作，对噪声是放大而非抑制的。

#### 层级敏感性

Dense Transformer 的层敏感性呈 **"U 形"**：
- 前 2-3 层：负责建立 token representation 的基础结构，误差被后续层放大
- 最后 2-3 层：直接影响 logit 输出，无后续层可以补偿
- 中间层：相对鲁棒，误差有机会在后续层中被"吸收"

Hessian trace（二阶敏感性指标）在首尾层最大，与上述观察一致。

#### Activation 分布特征

- **Massive Activation / [[Emergent Outlier Features]]**：特定 channel（通常 1-3 个）的 activation magnitude 是其他 channel 的 100-1000×，在所有 token、所有层中持续存在
- 导致 per-tensor 量化时大部分 channel 的有效 bit 只剩 2-3 bit（动态范围被 outlier 占据）
- Post-LayerNorm 的 activation 比 Pre-LayerNorm 更容易量化（LayerNorm 会 normalize 掉一部分 outlier）
- FFN 的 up-projection 输出（过 SiLU/GELU 后）分布不对称，有长尾

### MoE 模型（如 DeepSeek-V3 的 256 专家、Mixtral）

MoE 的量化困难**不在于**单个 expert 的 weight 分布（每个 expert 本身就是个小 FFN，分布和 Dense 类似），而在于**结构性异质**：

**Router（门控网络）的极端敏感性：**
- Router 输出是 softmax 前的 logits，通常很小（magnitude < 1），但微小扰动就会改变 top-K routing 决策
- 量化 router 哪怕只引入 0.01 的误差，都可能让 token 被路由到错误的 expert，造成级联错误
- 实验观察：router 做 INT8 量化后，routing agreement（与 FP16 的 routing 选择一致性）只有 85-90%
- **结论：Router 必须保留 FP16**

**Expert 间的分布异质性：**
- 高频 expert（频繁激活）：weight magnitude 较大，activation 分布稳定，校准数据充分 → 对量化相对鲁棒
- 低频 expert（冷门 expert）：weight magnitude 小但 variance 大，校准时几乎没有被激活 → 量化后可能完全失效
- Expert activation 频率呈幂律分布——top-20% 的 expert 处理了 80% 的 token

**校准不充分问题：**
- 对 Dense 模型 1024 条校准样本通常足够
- 对 256-expert MoE，某些 expert 在校准集中只被激活几次，其量化参数估计的方差极大
- 导致 MoE 量化的"长尾退化"——平均指标看着还行，但路由到冷门 expert 的特定输入质量严重下降

**选型指导：**
- Router → 保持 FP16
- 高频 expert → 可激进量化（W4），因为校准充分
- 低频 expert → 保守量化（W8）或增强校准数据
- MC-MoE (2024)：分层策略，区分 "always-activated expert" 和 "selective expert"

### 多模态模型（VLM，如 LLaVA/Qwen-VL）

多模态模型的量化需要**模态感知**——ViT、Projector、LLM 三个组件的敏感性分布截然不同。

**视觉 Encoder（ViT）——与 LLM 有本质不同的量化难点：**

- **Post-Softmax 分布**：Attention score 经 softmax 后集中在 [0,1] 且大部分接近 0，少量接近 1。这种极端不均匀分布对 uniform 量化非常不友好——大部分量化 level 浪费在 [0.1, 1.0] 区间，而真正需要精度的 [0, 0.05] 区间只分到 1-2 个 level
- **Post-GELU 分布**：输出有尖锐的 mode 在 0 附近加右侧长尾。和 ReLU 的简单 [0,∞) 截断不同，GELU 需要表示负值附近的微小差异
- **Channel outlier 不如 LLM 严重**：ViT 的 activation outlier 通常是 "spatial" 的（特定 patch 位置）而非 channel 维度的
- **Layer sensitivity 极不均匀**：第一个 patch embedding 层和最后的 CLS projection 层敏感性是中间层的 10 倍以上

**跨模态投影层（Projector）——信息瓶颈：**
- 通常是 1-2 层 MLP，将视觉 feature 映射到语言 embedding 空间
- 量化此层的误差同时丢失视觉信息 AND 产生对语言模型的分布外（OOD）输入
- Projector 做 W8A8 就会导致细粒度视觉理解能力（OCR、小物体识别）明显下降
- [[Q-VLM]] 发现视觉 encoder 的量化误差沿 pipeline 非线性累积（跨层依赖）

**语言 Decoder 部分：**
- 分布特征基本和纯 LLM 相同
- 额外现象：包含视觉 token 时，语言层 activation magnitude 可能更大（视觉信息能量更高）
- [[VLM Quantization Best Practices]] 发现量化方法本身改变重要性分布（AWQ 将 80-98% 重要性集中到 LLM）

**选型指导：**
- ViT → 最多 W8A8，关键层保持 FP16
- Projector → 建议不量化或仅 W8
- 语言 decoder → 同 Dense LLM 策略（W4A16 / W4A8）

### Diffusion 模型（Stable Diffusion / FLUX / DiT）

Diffusion 模型有一个独有维度——**时间步（timestep）**——使其成为所有架构中量化最特殊的：

**时间步依赖的 activation 分布剧变：**
- 去噪早期（t≈T）：activation magnitude 很大、variance 很高（处理粗结构）
- 去噪晚期（t≈0）：activation magnitude 变小但对精度极其敏感（精细细节）
- 同一层的 activation range 在不同 timestep 之间变化 10-50 倍
- **静态量化参数不可能同时适配所有 timestep**——这是 Diffusion 量化的核心矛盾

**U-Net 结构的特殊性：**
- Skip connection 将 encoder 的 activation 直接加到 decoder 对应层 → 量化误差不逐层衰减，反而通过 skip connection 被"注入"深层
- Up-sample 部分（重建细节）比 Down-sample 部分敏感 2-3 倍
- Cross-attention（条件注入文本 embedding）对量化极敏感——误差直接导致生成内容与 prompt 不对齐

**DiT 结构（FLUX/SD3）：**
- 没有 U-Net 的 skip connection 问题
- 但 AdaLN（Adaptive Layer Norm）的 scale/shift 参数对量化敏感——由 timestep embedding 生成，magnitude 变化大
- [[ViDiT-Q]] 发现 Static-Dynamic Channel Balance 是 DiT 量化的有效策略

**选型指导：**
- 必须用 timestep-aware 量化（如 TDQ, Q-Diffusion, [[ViDiT-Q]]）
- Cross-attention 层保留高精度
- W8 是安全下限，W4 需要按 timestep 分组

---

## 三、从问题到方法的映射

### 问题 1：Activation 存在极端 Outlier

这是 W8A8 / W4A8 的核心瓶颈。方法演进路线：

| 方法 | 核心思路 | 适用场景 | 局限 |
|------|---------|---------|------|
| [[SmoothQuant]] | per-channel scaling 将 activation 难度迁移到 weight | W8A8，outlier 中等 | 对极端 outlier 平滑不够 |
| OS+ / [[OmniQuant]] | 学习最优 scaling + shifting | W4A16~W4A8 | 需要 calibration |
| [[QuaRot]] / [[SpinQuant]] | 正交旋转消除 outlier 方向 | W4A4，outlier 严重 | 可能需要在线 Hadamard |

**选择依据**：如果模型 activation kurtosis > 10（outlier 极端），旋转基方法比缩放基方法更稳健，因为旋转是"能量均匀重分配"而缩放只是"通道间平衡"。

### 问题 2：要做 W4A4（追求极致计算加速）

4-bit activation 的 dynamic range 极有限，是当前最难的 PTQ 问题之一。

- [[QServe]]：QoQ 架构，per-token 动态量化 + SmoothAttention
- Atom (2024)：mixed-precision attention + reorder-based group quantization
- [[QuaRot]]：全模型旋转 + 在线 Hadamard 使 activation 更 uniform

**关键洞察**：纯 PTQ 做 W4A4 几乎必须配合 [[Rotation-based Quantization]]，否则精度崩溃。如果有训练资源，QAT（BitDistiller）效果显著更好。参见 [[Quantization Insights|Insight 8]] 中的最优 W4A4 配置。

### 问题 3：极限压缩（2-3 bit）

传统 scalar quantization（包括 GPTQ）在 2-bit 下失效——信息论意义上每个标量只能承载 4 个值，不足以表示 weight 的复杂分布。

| 方法 | 核心思路 | 理论基础 |
|------|---------|---------|
| [[QuIP#]] | [[Incoherence]] processing + E₈ [[Lattice Codebooks]] | 8 维最优球堆积，VQ |
| AQLM | Additive Quantization，多 codebook 加法组合 | 乘积量化理论 |
| GPTVQ (Google) | 类似 VQ，用于 Gemma | — |

**关键洞察**：2-bit 以下**必须**用 VQ/codebook 方法。代价是 decode 需要 codebook lookup，影响推理速度。

### 问题 4：MoE 模型量化

参见上方"MoE 模型"敏感性分析。核心策略：
- MC-MoE (2024)：按 expert activation 频率分层量化
- 高频 expert → W4（校准充分）
- 低频 expert → W8 或增强校准
- Router → FP16（绝对不能量化）
- Shared attention → 同 Dense LLM 策略

### 问题 5：PTQ 精度不够，是否需要 QAT

判断标准：如果 PTQ 后精度损失超过 1-2 个点（在你关心的 task metric 上），应该考虑 QAT。

| 方法 | 核心特点 | 计算代价 |
|------|---------|---------|
| LLM-QAT (Meta, 2024) | 合成数据做 QAT，避免对训练数据的依赖 | 中 |
| BitDistiller (2024) | 知识蒸馏 + QAT 联合，3-bit 接近 FP16 | 高 |
| EfficientQAT (2024) | Block-wise QAT 快速收敛 + E2E 精调 | 中低 |
| Google Gemma QAT | 训练时量化感知，发布 INT4 checkpoint | 已由 Google 完成 |

**新趋势**：轻量 QAT（几百步微调而非完整训练），如 EfficientQAT 的两阶段策略。

### 问题 6：KV Cache 量化

KV cache 和 weight 量化是**正交问题**，需要独立处理。核心原则见 [[KV Cache Quantization]]：

- Key → per-channel 量化（因为 Key 继承了 [[Emergent Outlier Features]] 的通道固定 outlier）
- Value → per-token 量化（因为 [[Attention Sparsity]] 为低权重 token 的量化误差提供天然保护）
- Pre-RoPE Key Quantization → 绕过 RoPE 对通道稳定性的破坏（[[KVQuant]]）

### 问题 7：多模态 / Diffusion 模型

参见上方相应架构的敏感性分析和选型指导。核心原则是**模态/组件感知的 mixed-precision**，而非统一量化。

---

## 四、决策树

```
你要部署什么模型？
├── Dense LLM (7B-70B)
│   ├── 能接受多少精度损失？
│   │   ├── 几乎零损失 → W8A8 (SmoothQuant / FP8)
│   │   ├── 轻微损失 (PPL +0.1~0.3) → W4A16 (GPTQ/AWQ)
│   │   ├── 中等损失可接受 → W4A8 / W4A4 (QuaRot + QServe)
│   │   └── 极限压缩 → W2-3bit (QuIP# / AQLM) + 考虑 QAT
│   └── 有训练资源吗？
│       ├── 有 → 先 PTQ 看精度，不够就 QAT（EfficientQAT / BitDistiller）
│       └── 没有 → 纯 PTQ 路线：Rotation + Scaling + GPTQ
├── MoE 模型
│   ├── Router → 保持 FP16
│   ├── 高频 expert → W4（标准 PTQ 方法）
│   └── 低频 expert → W8 或增强校准
├── VLM 多模态
│   ├── 视觉 encoder → W8A8 或更高
│   ├── Projector → W8 或不量化
│   └── 语言 decoder → 同 Dense LLM 策略
├── Diffusion 模型
│   ├── 必须用 timestep-aware 量化
│   ├── Cross-attention → 高精度
│   └── W8 安全下限，W4 需分 timestep
└── 长序列场景 (>32K)
    ├── Weight → 常规策略
    └── KV Cache → KIVI/KVQuant (Key:per-channel, Value:per-token) + eviction
```

---

## 五、当前未解决的问题

以下场景没有方法工作得很好，是开放研究方向：

**1. 长上下文精度退化**：量化模型在 4K 训练、128K 推理时精度退化速度远超 FP16。Needle-in-a-haystack 任务尤其脆弱。原因推测是 KV cache 中的量化误差沿序列长度累积。

**2. MoE Router 鲁棒性**：即使保持 FP16 router，量化后的 expert 输出分布变化仍会导致 routing 漂移（因为 router 的决策基于上一层的输出，而上一层已被量化）。缺乏理论分析和系统修复。

**3. 累积量化误差的理论界**：当前缺乏对"多层量化误差如何传播和累积"的 tight theoretical bound。实验表明早期层误差影响不成比例地大（与 U 形敏感性一致），但没有精确分析工具。

**4. Activation 4-bit 的通用方案**：尽管 [[Rotation-based Quantization]] 有帮助，W4A4 在 reasoning-heavy task（如 GSM8K、MATH）上精度仍不满意。[[Quantization Insights|Insight 8]] 显示 70B 模型的 W4A4 gap (+1.6) 反而大于 8B (+1.0)。

**5. 量化与 Safety Alignment 的交互**：初步研究发现量化可能破坏 RLHF 训练的 safety alignment，导致有害输出增加。机制不明，缺乏系统修复方法。

---

## 六、近期系统性选型研究

- **"A Comprehensive Evaluation of Quantization Strategies for LLMs" (2024)**：系统比较 GPTQ/AWQ/SqueezeLLM/OmniQuant 在不同 bit/model/task 组合下的表现，结论是没有 universal winner——最优方法取决于 model family + bit-width + task type
- **"Scaling Laws for Precision" (Harvard/Meta, 2024)**：建立模型规模-精度-性能的 scaling law。核心发现：更大模型对量化更鲁棒（相对精度损失更小）
- **"Which Quantization for Which Model?" (2025)**：基于模型 weight 分布特征（kurtosis, inter-channel variance, layerwise Hessian spectrum）的自动方法推荐系统
- **[[A Comprehensive Evaluation on Quantization Techniques for LLMs]]**：两步分解统一框架，9 条系统性 insight（见 [[Quantization Insights]]），是本 wiki 量化知识体系的核心骨架
- **[[PTQ-Bench]]**（HIT/IIT, 2025）：四策略分类（Compensation/Rotation/Salience/Optimization）+ 三维鲁棒性基准。关键结论：4-bit 选 AWQ，2-bit 欠训练模型选 QuIP，2-bit 充分训练模型选 GPTQ，跨结构（MoE/Mamba）选 GPTQ，追求最优鲁棒性选 Rotation+Compensation 组合。2-bit 大模型始终不如 4-bit 小模型。

---

## 七、总结对比：四类架构

| 维度 | Dense LLM | MoE | VLM | Diffusion |
|------|-----------|-----|-----|-----------|
| 核心难点 | Activation channel outlier | Expert 异质性 + Router 敏感 | 跨模态分布差异 | Timestep 动态 range |
| Outlier 类型 | Channel 维度固定 | Expert 间 scale 差异 | Spatial 维度 (ViT) | Temporal 维度 |
| 最敏感组件 | Embedding/lm_head + 首尾层 | Router + 冷门 Expert | Projector + ViT 首尾层 | Cross-Attn + 晚期 timestep |
| 静态量化可行性 | 较好 (per-channel) | 中等 (per-expert) | 分模态讨论 | 很差（需 timestep-aware） |
| 安全下限 | W4A16 | W4(高频) + W8(低频) | ViT:W8, LM:W4 | W8（需分 timestep） |

---

## 引用关系

← 理论基础：[[Emergent Outlier Features]]、[[Rotation-based Quantization]]、[[Quantization Insights]]、[[Quantization Granularity]]
← 部署对照：[[Quantization Deployment Formats]]（成熟框架选型）、[[Model Quantization Ecosystem]]（模型生态）
← 基准评估：[[PTQ-Bench]]（四策略鲁棒性基准）
← 架构特定：[[KV Cache Quantization]]、[[Q-VLM]]、[[VLM Quantization Best Practices]]、[[ViDiT-Q]]、[[Diffusion Model Quantization]]
→ 方法入口：[[GPTQ]]、[[AWQ]]、[[SmoothQuant]]、[[QuaRot]]、[[SpinQuant]]、[[QuIP#]]、[[QServe]]、[[OmniQuant]]
