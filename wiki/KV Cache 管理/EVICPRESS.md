---
title: "EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving"
aliases: [EVICPRESS, CacheServe, Joint Compression-Eviction]
source_type: paper
source_url: "https://arxiv.org/abs/2507.xxxxx"
source_date: 2025-07-01
ingested: 2026-05-13
venue: OSDI 2025 (or similar systems venue)
authors: [Shaoting Feng, Yuhan Liu, Hanchen Li, Xiaokun Chen, Samuel Shen, Kuntai Du, Zhuohan Gu, Rui Zhang, Yuyang Huang, Yihua Cheng, Jiayi Yao, Qizheng Zhang, Ganesh Ananthanarayanan, Junchen Jiang]
affiliation: University of Chicago / UC Berkeley / Tensormesh Inc. / MIT / UC Santa Cruz / Stanford / Microsoft
code: null
tags: [model-compression, kv-cache, eviction, compression, serving, system, multi-tier-storage, utility-function, adaptive, context-aware]
---

## 核心要点

EVICPRESS 是一个 KV cache 管理系统，核心创新在于**联合优化 compression 和 eviction 决策**——针对每个 context 的 KV cache，综合考虑压缩方法/比率选择和存储层级放置，以最大化整体 quality-TTFT trade-off。

![[images/EVICPRESS_fig1_tradeoff.png]]

关键洞察：

1. **不同 context 对压缩的敏感度差异巨大**（Insight 1）——有些 context 可以被激进压缩而不影响质量，有些即使极保守压缩也会崩溃
2. **最优压缩方法和比率因 context 而异**（Insight 2）——无单一方法/比率适用于所有 context，且与 context 长度无简单对应关系
3. **Compression 和 eviction 应联合决策**——单独压缩或单独 evict 都是次优的

结果：在 12 个 LongBench 数据集、5 个模型上，相同质量下 TTFT 降低 1.43×–3.77×（vs 固定压缩+LRU eviction），吞吐量提升 2.0×–3.6×（vs best baseline at 80% quality target），相比 full prefill 节省 1.29×–2.19× TTFT 且质量损失 <3%。

## 动机：为什么联合优化？

### 问题设定

在 KV cache reuse 的 serving 场景中（prefix caching），当 GPU 内存不足时，现有系统面临两类决策：
- **Eviction（即offloading）**：将 KV cache 从 GPU 迁移到 CPU DRAM / SSD / Remote Disk（经典 LRU）
- **Compression（即token dropping）**：压缩 KV cache 以在相同内存中存储更多 context（token dropping、quantization、merging 等）

现有系统分别独立优化这两类决策，错失了联合优化的巨大机会。

### 联合优化的收益

![[images/EVICPRESS_fig2_joint_example.png]]

考虑两个 KV cache：
- Context 1（4GB）：可压缩到 5% 大小而质量不变
- Context 2（8GB）：任何压缩都导致质量崩溃到 50%

| 策略 | Quality | TTFT |
|------|---------|------|
| Compression only（均压 50%） | 75%（低） | 0.3s |
| Eviction only（evict 半到慢存储） | 100% | 2.4s（高） |
| **Joint（压缩 Context 1 + 保留 Context 2）** | **100%** | **0.5s** |

联合策略将**不敏感** context 激进压缩释放空间，把**敏感** context 保持在快存储上无损——同时获得高质量和低延迟。

### 两大经验洞察

**Insight 1：Context 间压缩敏感度差异巨大**

在 Llama-3.1-8B-Instruct 上用 keydiff 压缩（ratio=0.9）：
- Wikipedia articles（2wikimQA）：中位敏感度 0.681（高敏感）
- Story narratives（NarrativeQA）：中位敏感度 0.340（低敏感，因对话冗余多）

→ 统一压缩方案会在敏感 context 上产生严重质量损失。

**Insight 2：最优方法/比率与 context 类型和长度无简单关系**

- 40.3% context 最优方法为 kvzip，40.0% 为 knorm，19.7% 为 snapkv
- 三组 context 的长度分布高度重叠 → 无法仅从长度/类型预测最优方法
- 即使同一数据集内，不同 context 的最优压缩比率也有很大标准差

## 方法：EVICPRESS 系统设计

![[images/EVICPRESS_fig6_workflow.png]]

### Utility Function：统一度量 Quality-Delay Trade-off

为每个 context 的每种 eviction-compression 配置定义单一得分：

```
Util(method, ratio, device) = (α · quality - TTFT) · frequency
```

- **α**：quality vs TTFT 的权衡参数（大 α → 偏好高质量，小 α → 偏好低延迟）
- **quality**：该配置下 compressed KV cache 生成答案与 uncompressed 答案的语义相似度
- **TTFT**：从存储设备加载 KV cache + prefill 新 query 的延迟
- **frequency**：该 context 的访问频率

以 **context-level**（而非 chunk-level 或 token-level）计算 utility，因为 context-level 拥有全局 important token 分布视图。

### 两阶段工作流

**Initial Offline Phase：**
1. 为每个 context 用初始 query 集计算所有可能配置的 utility score
2. 配置空间 = #compression_methods × #ratios × #storage_devices

**Online Inference Phase：**
1. 用户请求到来 → Lookup module 查找 KV cache 是否存在
2. 存在 → Retriever 从对应存储层加载 KV cache
3. 不存在 → 执行 full prefill，将 KV cache 存入 remote disk
4. 当存储层满时 → Configuration Selection Module 选择全局最优的 eviction/compression 方案

### Profiling Module：确定最优配置

- 使用 profiling query 集评估每个 context 在不同压缩方法/比率下的 quality
- **Periodic re-profiling**：当在线质量低于 profiled 质量阈值 X% 时触发
- Re-profiling 仅需压缩 KV cache + 运行 decoding phase → 计算开销小，可与 inference batch 混合
- 系统在 GPU 有空闲 cycle 时执行 re-profiling

### Configuration Selection Module：全局优化放置

当存储 tier S 满时，需更新已有 KV cache 的配置：
- 每个 context 可选择：更激进压缩留在 S，或 evict 到更低 tier
- 选项数 = #methods × #ratios × #lower_tiers
- 过滤掉使 KV cache 变大的选项
- 贪心选择使总 utility 最大化的配置组合

### 支持的压缩方法

EVICPRESS 支持多种 token dropping 方法作为压缩引擎：
- **keydiff**：计算每个 token 与所有其他 token 的平均 cosine similarity，丢弃相似度高的
- **knorm**：计算 Key cache 的 L2 norm，丢弃 norm 低的
- **snapkv**：计算 query 附近 token 与早期 token 的 cross-attention，丢弃低 attention 的

可扩展支持 KV cache quantization（如 CacheGen adaptive quantization、KIVI uniform quantization）。

## 实验结果

![[images/EVICPRESS_fig7_results.png]]

### 主要性能（5 模型 × 12 数据集）

| 对比 | TTFT 改进 | 质量改进 |
|------|----------|---------|
| vs 固定压缩 + LRU eviction | **1.43×–3.77× TTFT↓**（同质量） | +13.58%–55.40%（同TTFT） |
| vs Full prefill | **1.29×–2.19× TTFT↓** | <3% 质量损失 |
| vs Eviction only (LRU) | **1.22×–1.56× TTFT↓** | <3% 质量损失 |
| vs IMPRESS | **1.5×–5.2× TTFT↓** | +14.29%–27.00%（同TTFT） |

### Throughput（80% quality target）

- EVICPRESS 比 best baseline 高 **2.0×–3.6× QPS**
- 在同 ITL 下高 **1.9×–3.4× QPS**

### 关键发现

**配置多样性验证**：系统确实使用了广泛组合的配置——不同 context 被分配不同方法（keydiff/knorm/snapkv/no compression）和不同比率（0.1–0.4），验证了 per-context 自适应的必要性。

**Periodic re-profiling 效果**：
- 质量随时间逐步提升，最终 +11%（vs 无 re-profiling）
- Re-profiling 期间有短暂延迟峰值，但很快恢复
- 总体收益远超开销

**Per-dataset 改进**：
- 短 context（multi_news、qasper）改进最大
- 11/12 数据集上 EVICPRESS 优于所有 baseline
- 唯一例外 musique 略低于 keydiff+LRU

## 系统实现

- 基于 **vLLM v0.11.2** + **LMCache** 实现
- 多层存储：GPU → CPU DRAM (80GB) → SSD (800GB) → Remote Disk (unlimited)
- Profiling 和 re-profiling 非阻塞运行（CPU 上执行存储操作，GPU 可 batch re-profiling decoding）
- 系统开销极小：profiling 可与 decode batch 混合

### 支持的模型

- Dense：Llama-3.1-8B-Instruct, Qwen2.5-14B-Instruct, Mistral-7B-Instruct-v0.3, longchat-7b-v1.5-32k
- MoE：Qwen3-30B-A3B-Instruct-2507

## 与其他方法的定位

| 维度 | EVICPRESS | DiffKV | CAKE | SnapKV |
|------|-----------|--------|------|--------|
| **优化层次** | 系统级（跨 context 全局优化） | 系统级（per-head/request） | 算法级（per-layer） | 算法级（per-head） |
| **压缩 vs eviction** | 联合优化 | 联合（量化+pruning） | 仅 eviction | 仅 eviction |
| **Multi-tier 存储** | 是（GPU/CPU/SSD/Remote） | 否（仅 GPU） | 否 | 否 |
| **Per-context 自适应** | 是（方法+比率+设备） | Per-request（阈值统一） | 是（per-layer budget） | 否 |
| **Utility function** | α·quality - TTFT | 无 | Preference score | 无 |
| **Reuse 场景** | KV cache prefix caching | Online serving | Inference | Inference |
| **核心优势** | 跨 context 最优 quality-delay | Token 级差异化 | Layer 级自适应 | Observation window |

EVICPRESS 的独特定位在于**serving 系统级的跨 context 全局优化**——它不发明新的压缩算法，而是在已有方法（keydiff/knorm/snapkv）之上，做**per-context 最优配置选择 + 跨 tier 最优放置**，将不同方法的优势在不同 context 上发挥到极致。

## 局限性

- 当所有 KV cache 能 fit 在快存储中时，联合优化优势消失
- 当快/慢存储带宽差异不大时，eviction 优化空间有限
- 当前仅支持 token dropping 类压缩，quantization 方法（KIVI 等）尚未集成（但设计上可扩展）
- 单节点评估，多节点 / 多租户扩展为 future work

## 关键公式

```
# Utility Function
Util(method, ratio, device) = (α · quality - TTFT) · frequency

# Quality = cosine similarity between:
#   - embedding of answer generated with compressed KV cache
#   - embedding of answer generated with uncompressed KV cache
# using MiniLM-L6-v2 encoder

# TTFT = KV cache loading time from device + prefill time for new query
# frequency = access frequency of the context
```

## 相关链接

- 相关方法：[[Token Eviction]]、[[KV Cache Quantization]]、[[SnapKV]]、[[KeyDiff]]、[[KIVI]]、[[DiffKV]]
- 系统对比：[[QServe]]、[[Atom]]
- 综述：[[overview]]
