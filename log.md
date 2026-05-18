# Log

操作日志，按时间顺序记录每次 ingest、query、lint。

格式：`## [YYYY-MM-DD] 操作类型 | 描述`

---

## [2026-04-20] init | Wiki 初始化

- 创建目录结构：`raw/`, `raw/assets/`, `wiki/`
- 创建 Schema 文件：`AGENTS.md`
- 创建索引：`index.md`
- 创建日志：`log.md`（本文件）
- 创建总览页：`wiki/overview.md`
- Wiki 主题：LLM Quantization（大模型量化）
- 设计模式参考：[Karpathy LLM Wiki Pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)

## [2026-04-20] ingest | 批量录入 4 篇经典 PTQ 论文

**来源：**
1. **LLM.int8()** — Dettmers et al., NeurIPS 2022 — https://arxiv.org/abs/2208.07339
2. **GPTQ** — Frantar et al., ICLR 2023 — https://arxiv.org/abs/2210.17323
3. **SmoothQuant** — Xiao et al., ICML 2023 — https://arxiv.org/abs/2211.10438
4. **AWQ** — Lin et al., MLSys 2024 Best Paper — https://arxiv.org/abs/2306.00978

**创建的页面（11 个）：**

来源摘要（4）：
- `wiki/sources/LLM.int8().md`
- `wiki/sources/GPTQ.md`
- `wiki/sources/SmoothQuant.md`
- `wiki/sources/AWQ.md`

概念页（4）：
- `wiki/concepts/Post-Training Quantization.md`
- `wiki/concepts/Emergent Outlier Features.md`
- `wiki/concepts/Weight-Only Quantization.md`
- `wiki/concepts/Equivalent Scaling Transformation.md`

实体页（2）：
- `wiki/entities/MIT Han Lab.md`
- `wiki/entities/bitsandbytes.md`

对比页（1）：
- `wiki/comparisons/四篇经典 PTQ 方法对比.md`

**更新的页面（2 个）：**
- `wiki/overview.md` — 从空白填充为完整知识地图
- `index.md` — 添加全部 11 个新页面条目

## [2026-04-20] ingest | 录入综述论文：A Comprehensive Evaluation on Quantization Techniques for LLMs

**来源：**
- Liu et al., arXiv 2507.17417v3 — https://arxiv.org/abs/2507.17417
- 同济大学 / 布里斯托大学
- 综述/评测性质，在 W4A4 同一条件下公平对比各方法

**创建的页面（4 个）：**

来源摘要（1）：
- `wiki/sources/A Comprehensive Evaluation on Quantization Techniques for LLMs.md`

概念页（3）：
- `wiki/concepts/Rotation-based Quantization.md` — 正交旋转预量化变换
- `wiki/concepts/FP4 Quantization.md` — MXFP4/NVFP4 新数据格式
- `wiki/concepts/Quantization Granularity.md` — 量化粒度的性能-开销权衡

**更新的页面（4 个）：**
- `wiki/concepts/Post-Training Quantization.md` — 新增 W4A4 分支和两步分解框架
- `wiki/concepts/Equivalent Scaling Transformation.md` — 新增与 Rotation 的关系、FP4 下的表现
- `wiki/overview.md` — 重构为含两步分解框架的完整知识地图
- `index.md` — 添加 4 个新页面条目

## [2026-04-20] ingest | 录入两篇旋转优化量化论文：SpinQuant + QuIP#

**来源：**
1. **SpinQuant** — Liu et al. (Meta), ICLR 2025 — https://arxiv.org/abs/2405.16406
   - 发现随机旋转引入巨大方差（13 分），提出 Cayley SGD 在 Stiefel 流形上学习最优旋转
2. **QuIP#** — Tseng et al. (Cornell RelaxML), ICML 2024 — https://arxiv.org/abs/2402.04396
   - Randomized Hadamard Transform + E₈ lattice codebook，2-bit 极端压缩 SOTA

**创建的页面（4 个）：**

来源摘要（2）：
- `wiki/sources/SpinQuant.md`
- `wiki/sources/QuIP#.md`

概念页（1）：
- `wiki/concepts/Lattice Codebooks.md` — E₈ 格码本向量量化

实体页（1）：
- `wiki/entities/Cornell RelaxML Lab.md` — QuIP/QuIP# 来源实验室

**更新的页面（2 个）：**
- `wiki/concepts/Rotation-based Quantization.md` — 新增 Incoherence 理论、SpinQuant 方差分析、旋转位置/开销表、VQ 协同
- `index.md` — 添加 4 个新页面条目，来源数 5→7，总页面数 16→20

## [2026-04-20] ingest | 录入 3 篇新论文 + 创建优化/未优化变换概念体系

**来源：**
1. **QuaRot** — Ashkboos et al. (ETH/EPFL/IST Austria), NeurIPS 2024 — https://arxiv.org/abs/2404.00456
   - Computational Invariance + 随机 Hadamard，零数据端到端 W4A4KV4
2. **OSTQuant** — Hu et al. (Houmo AI/南大/东南), arXiv 2025 — https://arxiv.org/abs/2501.13987
   - QSUR 理论 + Riemann Adam 联合优化正交+缩放变换对
3. **FlatQuant** — Sun et al. (华为/清华/港中文), ICML 2025 — https://arxiv.org/abs/2410.09426
   - Kronecker 分解可学习仿射变换 + 缩放 + clipping，LLaMA-3-70B W4A4 ≤1% drop

**创建的页面（7 个）：**

来源摘要（3）：
- `wiki/sources/QuaRot.md`
- `wiki/sources/OSTQuant.md`
- `wiki/sources/FlatQuant.md`

概念页（4）：
- `wiki/concepts/Unoptimized Rotation.md` — 随机/固定旋转（QuIP#, QuaRot），零数据但有方差
- `wiki/concepts/Optimized Rotation.md` — 可学习旋转（SpinQuant, OSTQuant, FlatQuant），Stiefel 流形优化
- `wiki/concepts/Unoptimized Scaling.md` — 公式/搜索缩放（SmoothQuant, AWQ），快速鲁棒
- `wiki/concepts/Optimized Scaling.md` — 梯度优化缩放（OSTQuant, FlatQuant），与旋转联合最优

**更新的页面（4 个）：**
- `wiki/concepts/Rotation-based Quantization.md` — 新增指向 Unoptimized/Optimized Rotation 子页面的链接
- `wiki/concepts/Equivalent Scaling Transformation.md` — 新增"优化 vs 未优化缩放"节和子页面链接
- `wiki/overview.md` — 新增 2×2 分类矩阵、来源数更新
- `index.md` — 添加 7 个新页面条目，来源数 7→10，总页面数 20→27

## [2026-04-20] ingest | 创建 Quantization Insights 综合页面

**动机：** 将综述论文中具有指导意义的实验发现系统整理为独立词条，统合不同实验的 insights。

**创建的页面（1 个）：**
- `wiki/concepts/Quantization Insights.md` — 9 个核心 insights + 3 个元原则

**更新的页面（4 个）：**
- `wiki/concepts/FP4 Quantization.md` — 新增"FP4 时代需要什么预处理"开放问题
- `wiki/concepts/Quantization Granularity.md` — 新增与 FP4 的交互关系
- `wiki/concepts/Post-Training Quantization.md` — 新增"对称 vs 非对称量化"实验结论
- `wiki/concepts/Unoptimized Rotation.md` — 新增 Insight 2 交叉引用
- `index.md` — 添加 1 个新页面条目，总页面数 27→28
