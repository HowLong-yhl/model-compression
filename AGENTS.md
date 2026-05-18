# LLM-Wiki Schema: 量化 (LLM Quantization)

This file defines the conventions, structure, and workflows for maintaining this wiki. Any LLM agent operating on this vault should read this file first.

## Scope

This wiki is a **personal research knowledge base** focused on **LLM Quantization** — the theory, methods, tooling, and practical considerations for compressing large language models through quantization techniques (e.g. GPTQ, AWQ, GGUF/llama.cpp, SmoothQuant, QuIP, etc.).

## Language Convention

- **中英混合**: 正文以中文撰写，专业术语、方法名、模型名保留英文原文。
- 页面标题：概念/方法页用英文（如 `GPTQ.md`），综述/比较类页面用中文（如 `量化方法对比.md`）。
- 引用来源时保留原文标题（英文论文用英文标题）。

## Directory Structure

```
.
├── AGENTS.md          # 本文件 — Wiki schema 与工作流定义
├── index.md           # 内容索引 — 按分类列出所有 wiki 页面
├── log.md             # 操作日志 — 按时间记录 ingest/query/lint
├── raw/               # 原始来源（不可修改）
│   ├── assets/        # 图片与附件
│   └── *.md           # 源文档（文章、论文笔记等）
├── wiki/              # LLM 生成的 wiki 页面
│   ├── overview.md    # Wiki 总览
│   ├── entities/      # 实体页（模型、工具、组织）— 按需创建
│   ├── concepts/      # 概念页（量化方法、理论）— 按需创建
│   ├── comparisons/   # 对比与分析页 — 按需创建
│   └── sources/       # 来源摘要页
└── .obsidian/         # Obsidian 配置
```

> 子目录（entities/, concepts/, comparisons/, sources/）在首次需要时由 LLM 创建，无需提前建立空目录。

## Page Conventions

### Frontmatter

每个 wiki 页面应包含 YAML frontmatter：

```yaml
---
title: 页面标题
aliases: [别名1, 别名2]
tags: [quantization, method]   # 可选，用于 Dataview 查询
created: 2026-04-20
updated: 2026-04-20
sources: [来源1, 来源2]       # 关联的 raw/ 文件名
---
```

### Cross-references

- 使用 Obsidian `[[wikilink]]` 语法建立页面间链接。
- 首次提及一个有独立页面的概念/实体时，加上链接：`[[GPTQ]]`。
- 如果提及的概念尚无页面，仍然加上链接（Obsidian 会显示为红色，方便后续 lint 发现缺页）。

### Source Summary Pages

每篇 ingest 的来源在 `wiki/sources/` 下生成一个摘要页，格式：

```markdown
---
title: "论文/文章标题"
source_type: paper | article | blog | video | book_chapter
source_url: "https://..."
source_date: 2026-01-15
ingested: 2026-04-20
tags: [quantization, ...]
---

## 核心要点

- ...

## 方法/发现摘要

...

## 与已有知识的关联

- 与 [[相关页面]] 的关系：...
- 与 [[另一页面]] 的矛盾/补充：...

## 引用信息

原始来源：`raw/文件名.md`
```

## Workflows

### 1. Ingest（录入来源）

当用户提供新的来源文档时：

1. **阅读**来源全文，理解核心内容。
2. **与用户讨论**关键要点和值得注意的地方。
3. **创建来源摘要页** → `wiki/sources/` 下。
4. **更新或创建**相关概念页、实体页（`wiki/concepts/`, `wiki/entities/`）。
5. **更新** `wiki/overview.md`（如果新来源带来了对整体理解的重要补充）。
6. **更新** `index.md`（添加新页面条目）。
7. **追加** `log.md`（记录本次 ingest）。

一次 ingest 通常会涉及 5-15 个文件的创建或更新。

### 2. Query（查询）

当用户提问时：

1. 先读 `index.md` 定位相关页面。
2. 读取相关 wiki 页面，综合回答。
3. 回答应附带 `[[页面链接]]` 形式的引用。
4. 如果回答本身有长期价值（对比分析、深度解读），建议将其存为新的 wiki 页面。

### 3. Lint（维护检查）

定期或用户要求时进行：

- 查找页面间的矛盾。
- 识别孤立页面（无入链接）。
- 发现被引用但不存在的页面（红色链接）。
- 检查过时信息（已被更新来源取代的旧结论）。
- 建议补充的来源或概念。

## Category Tags

以下是本 wiki 的核心标签体系（随内容增长可扩展）：

- `quantization` — 通用量化相关
- `method` — 量化方法/算法
- `tool` — 工具与框架
- `benchmark` — 评测与对比
- `theory` — 理论基础
- `hardware` — 硬件相关优化
- `deployment` — 部署与推理
- `training` — 训练时量化（QAT 等）
- `post-training` — 训练后量化（PTQ）
