# 模型加速与压缩 研究 Wiki

一个关于**大语言模型加速与压缩（LLM Acceleration & Compression）**的个人研究知识库，基于 Obsidian 构建，通过持续录入来源文献逐步形成完整的知识图谱。

## 目标

1. **系统追踪**模型压缩、KV cache 管理、剪枝、架构效率等领域的前沿方法与理论演进
2. **建立关联**——通过 wikilink 将散落的论文知识点串联为可导航的概念网络
3. **提炼洞察**——从单篇论文的发现中归纳跨方法的共性原理与设计准则
4. **指导研究**——为自身研究方向选择提供系统化的 landscape 参考

## 覆盖领域

- **量化（Quantization）**：权重量化（PTQ/QAT）、激活量化（W+A）、KV Cache 量化、扩散模型量化、VLM 量化、注意力计算量化
- **KV Cache 管理**：KV Cache 量化、Token Eviction、Low-Rank 压缩与 Offloading、架构级 KV 压缩（GQA/MLA）、VLM KV Cache
- **剪枝与稀疏化**：非结构化剪枝、结构化剪枝
- **架构级效率改进**：GQA、MLA 等

## 目录结构

```
wiki/
├── README.md                        ← 本文件
├── overview.md                      ← 全局总览与知识体系导航
├── entities/                        # 跨领域研究机构（2 页）
│   ├── MIT Han Lab.md
│   └── Cornell RelaxML Lab.md
├── 量化/
│   ├── 理论与框架/                   # 核心概念、预量化变换理论、方法选型（17 页）
│   ├── 方法/                        # 具体量化方法论文（14 页）
│   ├── 多模态与扩散/                 # VLM、扩散模型量化（5 页）
│   ├── 注意力计算量化/               # SageAttention 系列（4 页）
│   ├── 部署与生态/                   # 部署格式、模型生态、工具（5 页）
│   └── 综述与对比/                   # 综述论文、方法对比（4 页）
├── KV Cache 管理/                    # KV cache 量化/eviction/low-rank/offload（31 页）
│   └── VLM KV Cache/               # VLM 专用 KV Cache 管理（5 页）
├── 剪枝与稀疏化/                     # 非结构化/结构化剪枝（3 页）
└── 架构级效率改进/                    # GQA、MLA 等（3 页）
```

## 页面格式约定

每个 `.md` 页面遵循统一格式：

```yaml
---
title: 页面标题
aliases: [别名1, 别名2]
tags: [标签1, 标签2]
created: YYYY-MM-DD
updated: YYYY-MM-DD
sources: [来源论文1, 来源论文2]
---
```

- **Frontmatter**：YAML 格式，包含标题、别名、标签、创建/更新日期、来源论文
- **Wikilink**：使用 `[[Page Name]]` 语法建立页面间关联
- **来源追溯**：每个事实标注来源论文，确保可验证性

## 核心知识体系

本 wiki 的量化部分以**两步分解框架**为骨架组织：

```
所有 PTQ 方法 = Step 1（预量化变换）+ Step 2（量化误差补偿）

Step 1: Rotation / Scaling（消除 outlier、平衡分布）
Step 2: RTN / GPTQ / Low-rank（补偿量化引入的误差）
```

KV Cache 管理部分以**四大正交子方向**组织：

```
总 KV cache ∝ (层数) × (KV head 数) × (token 数) × (每 entry 比特数)
              架构级 GQA/MLA   Token Eviction   KV Cache 量化
                                  ↕ Low-Rank/Offload
```

## 当前规模

- 已录入来源论文：56 篇
- Wiki 页面总数：95 页
- 上次更新：2026-05-18

## 使用方式

1. 使用 [Obsidian](https://obsidian.md/) 打开本目录作为 vault
2. 从 `overview.md` 开始导航全局知识体系
3. 使用 Graph View 查看概念间的关联网络
4. 通过标签（tags）按主题筛选页面

## 入口页面推荐

| 入口 | 说明 |
|------|------|
| `overview.md` | 全局总览，知识体系导航 |
| `量化/理论与框架/Quantization Method Selection.md` | 量化方法选型指南 |
| `KV Cache 管理/KV Cache Management.md` | KV cache 管理总览 |
| `KV Cache 管理/KV Cache Empirical Observations.md` | KV cache 经验观测汇总 |
| `量化/理论与框架/Quantization Insights.md` | 量化关键发现与趋势 |
| `量化/部署与生态/Quantization Deployment Formats.md` | 部署格式选型 |
