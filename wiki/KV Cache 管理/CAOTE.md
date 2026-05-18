---
title: "CAOTE: KV Cache Selection for LLMs via Attention Output Error-Based Token Eviction"
aliases: [CAOTE, FastCAOTE, Cache Selection via Attention Output Error]
source_type: paper
source_url: "https://arxiv.org/abs/2504.14035"
source_date: 2025-10-05
ingested: 2026-05-15
venue: arXiv 2025 (v6)
authors: [Raghavv Goel, Junyoung Park, Mukul Gagrani, Dalton Jones, Matthew Morse, Harper Langston, Mingu Lee, Chris Lott]
affiliation: Qualcomm AI Research
tags: [model-compression, kv-cache, eviction, token-eviction, attention-output, value-aware, meta-heuristic, inference]
---

## 核心要点

CAOTE 是首个以**闭合形式**将 attention score 和 value 向量统一整合为单一 eviction 评分的 token eviction 方法。核心洞察：现有 eviction 方法（H2O、TOVA、SnapKV 等）仅依赖 attention score（query-key 对齐度）作为 token 重要性代理，但 attention output 是 attention score 与 value 向量的**加权线性组合**——忽略 value 的贡献导致次优 eviction 决策。

CAOTE 直接优化 eviction 误差——eviction 前后 attention output 的均方误差——并证明该误差可以闭合形式精确计算（Theorem 3.2），无需实际执行 eviction：

$$c_j = \frac{\alpha_j}{1 - \alpha_j} \|V A^\top - v_j\|_2$$

其中 $\alpha_j$ 是 token $j$ 的 attention score，$VA^\top$ 是完整 attention output，$v_j$ 是 token $j$ 的 value 向量。**evict CAOTE score 最低的 token**——即对 attention output 影响最小的 token。

关键特性：CAOTE 是一个 **meta-heuristic**——可以叠加在任何基于 attention score 的 eviction 方法上，只需将原始 score 归一化到 [0,1] 后代入 CAOTE 公式。

![[images/CAOTE_fig1_method_flow.png]]
*Figure 1：CAOTE 方法流程。对每个 token 分别计算 eviction 后的 attention output 变化（即 CAOTE score $c_j$），evict 影响最小的 token。修改量极小——仅在现有 eviction pipeline 中增加 value 信息的整合。*

## 理论基础

### Theorem 3.1：Post-eviction attention score 的封闭关系

Evict token $j$ 后，其他 token $i$ 的 attention score 重分配为：

$$\alpha_i' = \frac{\alpha_i}{1 - \alpha_j}$$

这意味着 eviction 后 attention score 按**比例放大**，放大系数为 $1/(1-\alpha_j)$。

### Theorem 3.2：CAOTE score ≡ Eviction error

CAOTE score 与 eviction 误差（Eq.9 中的 MSE）**精确相等**：

$$c_j = e_{\text{eviction},j} = \|X_{\text{attn}} - X'_{\text{attn},j}\|_2$$

证明关键步骤：将 post-eviction attention output 代入得 $X'_{\text{attn},j} = \frac{1}{1-\alpha_j}(X_{\text{attn}} - \alpha_j v_j)$，展开差值即得 $c_j$。

### 与 output logits 的关系（Appendix D.2）

论文进一步证明 CAOTE score 与最终 logit 的变化成正比：
- 单层单头：$\Delta l = W_H W_O \Delta_A$，其中 $\Delta_A = e_{\text{eviction}}$
- 多头：$\Delta_A = \text{concat}(\Delta_A^1, ..., \Delta_A^h)$
- 含 FFN：误差通过 $W_{\text{FFN}} W_O$ 进一步放大

这意味着最小化 CAOTE score 直接有助于减少最终预测的偏差。

## Meta-Heuristic 用法

CAOTE 的核心创新在于它可以作为 **meta-method** 叠加在任何 attention-score-based eviction 方法上：

$$c_j = f_j^{\text{caote}}(f^{\text{norm}}(H), V) = \frac{h_j^{\text{norm}}}{1 - h_j^{\text{norm}}} \|V(H^{\text{norm}})^\top - v_j\|_2$$

**关键要求**：原始 eviction score 需归一化使其和为 1。不同方法的适配：

| 方法 | 原始 score | 归一化方式 | 备注 |
|------|-----------|-----------|------|
| TOVA | $\alpha_{-1,j}$（最后一个 query 的 attention） | 天然满足 $\sum = 1$ | 无需额外归一化 |
| H2O | $\sum_i A_{i,j}$（累积 attention） | 除以 $\sum_j h_j$ | 证明 $\sum h_j > 1$（Theorem D.1） |
| SnapKV | observation window 内 attention 投票 | 除以 $\sum_j h_j$ | 类似 H2O |

| 方法 | Keys | Values | 最小化 eviction error |
|------|------|--------|---------------------|
| H2O | ✓ | ✗ | ✗ |
| TOVA | ✓ | ✗ | ✗ |
| SnapKV | ✓ | ✗ | ✗ |
| X + CAOTE | ✓ | ✓ | ✓ |

## FastCAOTE：高效近似

为减少计算开销，FastCAOTE 将完整 attention output $X_{\text{attn}}$ 替换为 value 向量的均值：

$$c_j^{\text{fast}} = \frac{\alpha_j}{1 - \alpha_j} \left\| \frac{1}{b+1} \sum_{i=1}^{b+1} v_i - v_j \right\|_2$$

- 避免了计算完整 attention output 的矩阵乘法
- 与 full CAOTE 的 Spearman 相关性 **≥ 0.8**（每层，Table 8）
- 实际性能几乎等价甚至略优于 full CAOTE（见 LongBench 结果）
- 计算开销极低：prefill 阶段仅增加 $3.69 \times 10^{-5}$（32K）至 $8.9 \times 10^{-5}$（4K）的 FLOP 比例

## 实验设置

- **模型**：Llama 3.2-3B-Instruct, Llama 3.1-8B-Instruct, Qwen 2.5-3B-Instruct, Qwen 2.5-7B-Instruct
- **Baselines**：H2O, TOVA, SnapKV（各自 ± CAOTE/FastCAOTE）
- **Budget**：2K, 4K, 6K, 8K
- **Prompt 处理**：Block-wise prefill，block_size = 128（非一次性 prefill → 在 prefill 阶段就开始 eviction）
- **评估**：LongBench（16 tasks）、Booksum 困惑度、Needle-in-a-Haystack（NIAH）

## 主要结果

### LongBench

![[images/CAOTE_table2_longbench_llama.png]]
*Table 2：Llama 3.1-8B / 3.2-3B 在 LongBench 16 项任务上的结果。CAOTE/FastCAOTE 在几乎所有方法-budget 组合上一致提升性能。*

核心发现：

- **H2O 提升最大**：Llama 3.1-8B 2K budget，H2O 16.89 → H2O+FastCAOTE **34.07**（+17.18，超 100% 相对提升）。H2O 原本 2K 下大幅退化（只有 dense 的 34%），加 CAOTE 后恢复到 dense 的 69%
- **SnapKV+FastCAOTE 综合最优**：Llama3 系列最佳平均表现（2K: 40.73, 4K: 45.75）
- **Qwen 2.5 上 SnapKV+CAOTE 最优**：2K Qwen2.5-7B 29.24, 4K 33.19
- **所有 baseline 均受益**：无论 H2O、TOVA 还是 SnapKV，叠加 CAOTE/FastCAOTE 后平均精度一致提升
- **QA 类任务 H2O+CAOTE 特别强**：Qasper、MF-en、HotpotQA、2WikiMQA、Musique 上 H2O+CAOTE 常为最优

### Perplexity

![[images/CAOTE_table4_perplexity.png]]
*Table 4：Booksum 困惑度差异（相对 dense baseline）。负值表示优于 dense。Llama 3.1-8B dense PPL = 9.833，Llama 3.2-3B dense PPL = 15.4911。*

- **TOVA/SnapKV + CAOTE 的 PPL 低于 dense baseline**：Llama 3.1-8B 6K budget，TOVA+FastCAOTE **-0.085**（PPL 低于无 eviction 的 dense 模型！）
- H2O 的 PPL gap 被 CAOTE 大幅缩小：2K budget Llama 3.1-8B 从 2.007 降至 1.884（CAOTE）/ 1.891（FastCAOTE）
- Llama 3.2-3B 上所有方法 + CAOTE 的 PPL gap 一致缩小

### Needle-in-a-Haystack

![[images/CAOTE_fig2_needle_haystack.png]]
*Figure 2：Llama 3.1-8B 6K budget NIAH 热图。上排：baseline H2O / TOVA / SnapKV 的大面积红色（检索失败）。中排：+CAOTE 后显著绿化。下排：+FastCAOTE 效果相当。*

![[images/CAOTE_table5_niah.png]]
*Table 5：NIAH 精确 recall 数值。H2O+CAOTE 在 Llama 3.1-8B 上提升最大（4K: 0.330→0.538/0.568）。*

- **H2O 改善最显著**：4K budget Llama 3.1-8B，H2O 0.330 → H2O+FastCAOTE **0.568**（+72% 相对提升）
- **6K budget**：H2O 0.544 → H2O+CAOTE **0.698**（+28%）；SnapKV 0.490 → SnapKV+FastCAOTE **0.580**（+18%）
- Llama 3.2-3B 上提升更稳定但幅度较小，H2O+FastCAOTE 在所有 budget 下最优

## 局限性（Appendix A）

1. **Greedy/Myopic 策略**：CAOTE score 基于单 token eviction 假设。Prefill 阶段需同时 evict 多个 token（block_size = 128），但联合多 token eviction 的组合爆炸使精确计算不可行（$\binom{n}{m}$ 种组合）
2. **多 token eviction 的闭合形式**：虽然论文推导了联合 eviction score（Appendix B, Eq.31：$c_{[1,...,m]} = \frac{1}{1-\Sigma \alpha_i} \|\Sigma \alpha_i(X_{\text{attn}} - v_i)\|_2$），但实际仍需贪心逐 token 处理
3. **Block-wise prefill 约束**：方法假设 block-wise prompt consumption（block_size=128），实际部署中需验证与不同 prefill 策略的兼容性

## 研究意义

### 理论贡献

1. **首个 value-aware 闭合形式 eviction score**：此前所有方法要么纯 attention score，要么需启发式近似（如 VATP 的 attention × value norm）。CAOTE 直接从 eviction error 目标推导出精确闭合形式
2. **eviction error = CAOTE score 的等价证明**：建立了 eviction 优化目标与可计算评分之间的精确理论联系（而非 upper bound 或近似）
3. **Meta-heuristic 性质**：不与特定 eviction 方法绑定，而是提供了一个通用的 value 信息整合框架

### 与相关工作的关系

- **vs [[H2O]]**：H2O 用累积 attention score，CAOTE 在其基础上加入 value 距离；H2O+CAOTE 是 H2O 的严格升级
- **vs [[SnapKV]]**：SnapKV 用 observation window voting，CAOTE 为 vote 结果补充 value 信息；SnapKV+FastCAOTE 是 Llama3 最优组合
- **vs [[TOVA]]**：TOVA 直接用最后 query 的 attention（天然 $\sum=1$），CAOTE 叠加最自然
- **vs [[KeepKV]]**：KeepKV 改进 eviction **action**（eviction→merging），CAOTE 改进 eviction **scoring**（attention→attention+value）。两者正交互补
- **vs [[KeyDiff]]**：KeyDiff 完全放弃 attention score 用 Key 几何多样性，CAOTE 则在 attention 基础上增补 value 信息。理论出发点不同但都指向同一问题——纯 attention score 不足以评估 token 重要性
- **vs [[Expected Attention]]**：EA 将 value 影响力 $\|W_o v_i\|$ 作为乘法因子，CAOTE 用 $\|X_{\text{attn}} - v_j\|_2$ 衡量 value 偏离 attention output 的距离。两者都属于 "value-aware" 方法但数学形式不同

### 更广泛的 attention score 局限性

CAOTE 的成功再次验证了 **attention score 作为 token 重要性代理的根本局限**——一个被多篇独立工作从不同角度揭示的主题：

- **VATP**（EMNLP 2024）：attention × value norm
- **CAOTE**（本文）：attention output error 闭合形式
- **[[Expected Attention]]**：expected attention × $\|W_o v_i\|$
- **Ada-KV**（NeurIPS 2025）：theoretical upper bound 分析
- **CurDKV**（NeurIPS 2025）：Value-guided CUR 分解
- **[[MixKV]]**（ICLR 2026）：attention（importance）+ diversity 联合

## 引用关系

**本页引用**：
- [[H2O]] — CAOTE 的核心 baseline，meta-heuristic 的典型应用目标
- [[TOVA]] — 另一 baseline，score 天然 $\sum=1$ 无需归一化
- [[SnapKV]] — 第三个 baseline，SnapKV+FastCAOTE 为 Llama3 最优
- [[KeepKV]] — 正交互补（scoring vs action 改进）
- [[KeyDiff]] — 同 Qualcomm AI Research，不同方法路线（attention-free vs value-aware）
- [[Expected Attention]] — 另一 value-aware 方法（预测未来 attention × value 影响力）
- [[Token Eviction]] — CAOTE 属于 Token Eviction 子方向
- [[KV Cache Empirical Observations]] — value 向量的重要性是经验观测的一部分
