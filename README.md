# kb_rag —— 面向个人学习知识库的 **Agentic RAG**（占位名，可改）

> 全新项目，**独立于 （一个更早的项目）**。吸取 （一个更早的项目） 两个教训：
> ①「根本不 agentic」→ 本项目核心是 LLM 驱动的自主回路，不是写死管线。
> ②「定位/语料错」→ 只吃你自己的**学习知识库原始资料**（transcript + 书 PDF 文本），一个明确身份。

## 这是什么

一个**忠实的、agentic 的学习助手**，跑在你重抓+清洗的原始课程语料上，服务两个场景：

- **搜索**（`ask`）：对整个知识库准确问答，答案带原文引用、无支撑则拒答。
- **教学**（`teach`）：整节原文**有序取回 + 覆盖核查**，喂给苏格拉底教学，**保证不漏细节**。

## 为什么 agentic（对症 （一个更早的项目） 教训①）

分工照 BCA M08 黄金法则「能 workflow 就别 agent」：

```
确定性 workflow（代码，engine.py）：分块 / 检索 / RRF / 引用 verify —— 稳、可审计
Agentic 自主（LLM，agent.py）：① 规划 ② 迭代检索"够不够" ③ 综合 ④ 引用自查
```

回路：
```
question → plan（拆子问题/选 search|teach/路由课程）
        → retrieve-loop（调 search/fetch_section；judge 覆盖 → 不够就重写再搜 ↺）
        → synthesize（基于检索块起草，逐句挂引用）
        → self-check（verify 每条引用；无支撑 → 补搜或拒答）
        → Answer{answer, citations, coverage, blocked}
```
「不 agentic」的病根就是缺 ②④——本项目的自主性正好放在这两处，其余全确定性。

## 语料边界（对症 （一个更早的项目） 教训②）

- 只吃 `knowledge_base/raw/` 下**清洗过的原始资料**，打 `layer=raw`。
- NotebookLM 拆解产物**不进这个池**（它走"产出内容 + 自己看"另一条路），保持 raw 池忠实纯度。
- 每块带 `course / kind / section / order / layer` 元数据 → 支持按课/按节过滤与有序取回。

## 快速开始

```bash
# 1) 原始资料 → 语料（layer=raw）
python -m kb_rag.ingest --src knowledge_base/raw --out chunks.jsonl

# 2) 问（搜索 agent，默认 mock 后端，纯 stdlib）
python -m kb_rag.cli ask "temperature 怎么调"

# 3) 教（教学 agent，整节有序取回 + 覆盖核查）
python -m kb_rag.cli teach "监督学习" --course "Sample"

# 4) 真 LLM 后端（本地插 key 后）
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
python -m kb_rag.cli ask "..." --backend anthropic
```

## 后端：mock / 可插（照 （一个更早的项目） spec 保留的好做法）

- `MockBackend`：纯标准库、确定性，用启发式冒充 LLM 决策 → **离线即可测 agent 控制流**（本仓库测试就跑它）。
- `AnthropicBackend`：本地插 `ANTHROPIC_API_KEY` 激活即用真 LLM。四个决策点各发一条结构化 prompt 给 Claude（官方 `anthropic` SDK）：`plan`/`judge`/`selfcheck` 用结构化输出（`output_config.format`，低 effort），`synthesize` 纯文本、只依据检索材料生成；系统提示词固定在前利于 prompt cache，`stop_reason: "refusal"` 与解析失败都有兜底不阻断回路。模型默认 `claude-sonnet-5`，可用 `--model` 或 `KB_RAG_MODEL` 覆盖。

## 目录

```
kb_rag/
├── engine.py   确定性检索核 + 工具（search/fetch_section/outline/list_courses/verify）
├── ingest.py   原始资料(md/txt/vtt/srt) → chunks.jsonl（清洗字幕、分节、打元数据）
├── agent.py    后端(Mock/Anthropic) + agentic 回路（search / teach 两模式）
└── cli.py      命令行：ingest / ask / teach / courses
tests/
├── fixture/    最小样例语料（供 mock 冒烟）
└── test_smoke.py
```

## 现状 / 待办

- ✅ 确定性核 + agentic 回路（mock）端到端可跑；教学模式整节有序取回 + 覆盖核查。
- ⏭ 落地 AnthropicBackend 四个 prompt；raw 语料接入（你本地抓+清洗）；PDF→txt 抽取脚本。
- ⏭ 可选：索引持久化、增量加料、boundary 录制重放（要复现性再上，别过度工程）。

> ⚠️ 语料是第三方课程原文 → **仅本地私用，别公开**。`chunks.jsonl` 等构建产物走 `.gitignore`。
