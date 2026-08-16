# 检索与答案质量记录（Retrieval & Answer Scores Log）

这份文档**追加式**记录每次评测的分数，用于 review project 时对照效果、判断调参/改结构有没有起作用。
每次跑完把结果按日期追加到最下面，不要覆盖历史。

## 名词：三个分数分别在量什么

| 分数 | 来源 | 量的是 | 阈值/参考 |
|---|---|---|---|
| **cos**（检索层，语义相似） | `python -m kb_rag.scores` | 查询向量 vs 块向量 的余弦。反映"搜得准不准"。 | 越高越好；短查询 vs 长块，0.4~0.6 属正常 |
| **cov**（检索层，词面覆盖） | `python -m kb_rag.scores` | 查询词有多少比例出现在块里（BM25 那一路）。 | 跨语言时会=0（词不重合），属正常 |
| **veto 语义核对分**（答案层） | `ask` 输出的 `· veto: 语义核对 score=` | 最终**答案** vs 被引**原文块** 的余弦。防幻觉的关卡。 | 地板 `SEM_FLOOR=0.30`，低于即拒答 |

关键认知（来自实测，见下方 2026-08-09 记录）：
- **veto 分主要被"改写/综合"拉低，而不是"跨语言"。** 答案被重组成分点总结时，中英文都会掉到 ~0.70。
- **跨语言检索时词面 `cov=0`，全靠向量在扛** → 所以必须混检、必须加大 `W_VEC`、防幻觉必须用 `verify_semantic` 而非词面 `verify`。

---

## 当前调参基线（复现实验需一致）

```
KB_RAG_LEX_MIN_COV=0.5     词法命中覆盖低于此值丢弃（防跨语言词法垃圾）
KB_RAG_W_LEX=1.0           词法在 RRF 里的权重
KB_RAG_W_VEC=2             语义在 RRF 里的权重（跨语言加大）
KB_RAG_SEM_FLOOR=0.30      答案语义支撑地板（veto）
KB_RAG_MAX_STEPS=12        agent 回路步数上限
embedder = OpenAIEmbedder (text-embedding-3-small)
LLM      = OpenAI (agentic-openai)
```

---

## 2026-08-09 · 基线评测（HITL / 人在回路）

**语料状态**：course/book/video 已入库；书籍尚未分章（每本书 = 一整块 section）。

### 检索层 `scores`（同一主题，中英对照）

命令：
```
python -m kb_rag.scores "how does human in the loop approval work" "什么是人在回路 怎么加审批"
```

英文查询 `how does human in the loop approval work`：

| rrf | cos | cov | 来源 |
|---|---|---|---|
| 0.0497 | 0.551 | 0.75 | [course] Building AI Research Agents · C_34 Human-in-the-Loop |
| 0.0487 | 0.537 | 0.75 | [course] Building AI Research Agents · C_34 Human-in-the-Loop |
| 0.0457 | 0.499 | 0.75 | [book] Agentic-Design-Patterns · Agentic-Design-Patterns |
| 0.0452 | 0.524 | 0.75 | [video] AI Engineer · (youtube) |
| 0.0448 | 0.489 | 0.62 | [course] Building AI Research Agents · C_34 Human-in-the-Loop |

中文查询 `什么是人在回路 怎么加审批`：

| rrf | cos | cov | 来源 |
|---|---|---|---|
| 0.0333 | 0.459 | 0.00 | [course] Building AI Research Agents · C_34 Human-in-the-Loop |
| 0.0328 | 0.429 | 0.00 | [course] Building AI Research Agents · C_38 KPI Cards |
| 0.0323 | 0.429 | 0.00 | [course] Building AI Research Agents · C_42 Reviewer |
| 0.0317 | 0.428 | 0.00 | [course] Building AI Research Agents · C_34 Human-in-the-Loop |
| 0.0312 | 0.427 | 0.00 | [course] Building AI Research Agents · C_35 End-to-End Test |

**对照小结**

| | 英文 | 中文 |
|---|---|---|
| cos（语义） | 0.49–0.55 | 0.43–0.46 |
| cov（词面） | 0.75 | **0.00** |
| top-1 命中 | ✅ C_34 | ✅ C_34 |
| 2–5 名 | 都在 HITL 正确区 | 有噪声（C_38/C_42/C_35） |

- 英文 cos 比中文高约 0.08–0.10（同语言匹配更紧），差距不大。
- 中文 `cov=0`：中文问题词在英文原文里不出现，词法零贡献，**全靠向量**。
- 中文 top-1 正确，但 2–5 名有噪声 = 跨语言判别力上限。

### 答案层 `ask`（veto 语义核对分）

| 问题 | 语言 | 轮次 | veto 分 | 引用 | 结果 |
|---|---|---|---|---|---|
| 什么是人在回路，怎么加审批 | 中 | 4 | **0.695** | C_34 | ✅ 准确、无幻觉 |
| How does human-in-the-loop approval work in an agent workflow | 英 | 8 | **0.705** | Agentic-Design-Patterns + C_34 | ✅ 准确、无幻觉 |

- 两条几乎相同（0.695 vs 0.705）→ 印证 veto 分主要被"改写/综合"拉低，跨语言不是主因。
- 地板 0.30，两条都有 ~0.4 富余 → 防幻觉稳。
- 英文那条跑满 8 轮（过度搜索），靠"最后一步强制收尾"才正常出答案 → 已把上限提到 12。

### 待改进（本次暴露的问题）

1. **书籍未分章**：`Agentic-Design-Patterns` 整本是一块，引用只能指到"整本书"。分章后 cos 更高、噪声更少、引用精确到章节。 → **已实现，见下方 2026-08-09 变更**
2. 中文检索 2–5 名噪声：靠 agent 的 `fetch_section` + `verify` 在端到端兜住（最终答案仍准），但检索层本身可通过分章 + 元数据 scoping 进一步降噪。 → 分章后应改善，待重嵌后复测。

---

## 2026-08-09 · 变更：书籍分章（页码分节）+ 块 id 稳定化

**改了什么**（`kb_rag/ingest.py`）：

1. **书籍按 `[p.N]` 页码窗口分节**。之前每本书 = 一整节（引用只能指"整本书"）；现在每 `--book-pages`（默认 8）页为一节，节名 `p.a-b`，尽量附章节提示（如 `p.312-319 · 第18章 …`）。选页码而非章节作锚点：语料多语言 + OCR，章节标记常被打花，页码标记 100% 覆盖（47/47 本非空书都有）且引用精确到页、便于回原书核对（正合防幻觉）。无页码标记的书退回整篇。
   - 实测：《思考，快与慢》1 节 → 61 节；《Agentic-Design-Patterns》1 节 → 61 节。全库书籍 distinct section 从 ~47 → 1573。
2. **块 id 改为 `md5(相对路径 | 文件内局部序号)`**，不再含全局 order。修掉一个潜在缺陷：之前 id 含跨文件全局序号，**加任何一个文件都会顺移后续所有 id → 增量嵌入整库重嵌**，架空了"只嵌新增"。现在加新书只产出新 id、老 id 全不变（有测试锁定 `tests/test_smoke.py::BookSectioning`）。

**⚠️ 需要重做一次（一次性）**：因为书籍 chunk 的 section 和 id 都变了，要重新 ingest + 重嵌。id 稳定化是全库一次性迁移，之后加数据才只嵌新增。建议起干净的向量库：

```
python -m kb_rag.ingest --src <你的RAG-Database路径> --out chunks.jsonl
python -m kb_rag.batchembed submit --chunks chunks.jsonl --out vectors.npz
（等几分钟~1小时后）
python -m kb_rag.batchembed fetch --out vectors.npz
```

（数据量大用上面的 Batch API；小改动可用 `python -m kb_rag.buildindex`。旧 `vectors.npz` 里的老 id 会变成无用残留，想干净可先删掉 `vectors.npz` 和 `vectors.npz.batch.json` 再重嵌。）

**前置要求**：书要放成 **每本一个文件夹** `book/<书名>/<书名>.txt`，这样 `course=书名`、引用读作《书名》· p.a-b。若平铺在一个占位文件夹（如仓库里的 `book/示例书名/`），所有书会挤在同一 course、页码节名会跨书撞名——你实际的库已是每本一文件夹（打分输出里 course=书名 可证），保持即可。

**重嵌后复测**：把本文最上面那两条命令（`scores` 中英对照 + 英文 `ask`）重跑，把新分数追加到下面，对比分章前后 cos / 噪声 / veto 分的变化。

---

## 2026-08-09 · 分章后复测（page-based sectioning 生效）

**语料状态**：书籍已按 `[p.N]` 页码分节；全库重嵌完成（同步 buildindex，53659 块）。

### 课程类问题（HITL）— 预期无变化，已验证无变化

同一条 `scores "how does human in the loop approval work" "什么是人在回路 怎么加审批"`，
两组分数与 8-09 基线**完全一致**。原因：命中全在课程 transcript（C_34 等），书没参与，
分章只改书籍切分 → 对课程类问题零影响（符合预期）。

一个正向信号：基线时英文第 3 名是**《Agentic-Design-Patterns》整本书**(cos 0.499)，
分章后它**掉出前 5**——整本书拆成页后，没有单页能在 HITL 问题上压过课程/视频片段。
即"整本书混进来"的噪声被消除（正确降噪）。

### 书籍类问题 — 分章价值显现

命令：`python -m kb_rag.scores "锚定效应是什么" "系统1和系统2有什么区别"`

Q `锚定效应是什么`：

| cos | cov | 来源（分章后精确到页/章） |
|---|---|---|
| 0.561 | 0.67 | [book]《思考，快与慢》· p.139-146 · 第12章 |
| 0.549 | **1.00** | [book]《思考，快与慢》· **p.131-138 · 第11章 锚定效应在生活中随处可见** |
| 0.548 | 0.50 | [book]《思考，快与慢》· p.139-146 · 第12章 |
| 0.490 | 0.67 | [book]《思考，快与慢》· p.131-138 · 第11章 |
| 0.530 | 0.83 | [book]《思考，快与慢》· p.131-138 · 第11章 |

Q `系统1和系统2有什么区别`：

| cos | cov | 来源 |
|---|---|---|
| 0.534 | 0.67 | [book]《思考，快与慢》· p.35-42 · 第2章 |
| 0.503 | 0.50 | [book]《思考，快与慢》· p.35-42 · 第2章 |
| 0.495 | 0.67 | [book]《思考，快与慢》· p.75-82 · 第6章 |
| 0.456 | 0.67 | [book]《思考，快与慢》· p.27-34 |
| 0.451 | 0.67 | [book]《思考，快与慢》· p.51-58 · 第4章 |

**结论**：
- 分章前书籍只能引"整本书"（`《思考，快与慢》· 思考，快与慢`），无页码、向量糊。
- 分章后精确到页码区间 + 章节名；"锚定效应"问题的第 2 名正是 `第11章 锚定效应`（cov=1.00），
  可直接翻到 p.131-138 核对。命中全落在书的正确章节区域。
- **分章的价值点确认：书籍类问题引用精确到页/章 + 检索更聚焦；课程类问题不受影响（本就不该受影响）。**

### 成本/部署备注（本轮）

- 全量重嵌因 Batch 排队 token 上限（本组织 3M < 全库 32M）失败，改用同步 `buildindex` 完成。
- 之后**加数据只嵌新增**（id 只依赖单文件，已修）；提问时推理走 Claude 订阅（0 API 费），
  仅问题/答案的 embedding 有可忽略开销。

---

## 黄金集评估（系统化，非抽查）

上面那些是**抽查**。要一个可重复、可报的数字，用黄金集评估 `kb_rag.eval`：

```
python -m kb_rag.eval --golden eval/golden.jsonl --k 8            # 用 vectors.npz 混检出真实数字
python -m kb_rag.eval --golden eval/golden.jsonl --verbose        # 逐题看命中排名
```

- 评的是**检索层**（确定性、免费、可重复）：正确来源有没有进 top-k。
- 指标：**Recall@1/3/5/k** + **MRR**，并按 语言(zh/en) 和 类型(book/course/video) 分组。
- 未命中的题会单列——那是最该修的检索问题（召回错 → 调分块/阈值/权重/rerank）。
- `eval/golden.jsonl` 是**种子集**（15 题），按需增删改成你自己的代表性问题。每条：
  `{"q": 问题, "lang": "zh|en", "expect": {"kind":..., "course":子串, "section_contains":子串}}`，
  `expect` 可为 list（命中任一即算对）。**黄金集质量决定评估质量**——照实写期望来源。

⚠️ 一定要带 `vectors.npz` 在**你 Mac 上**跑才是真实数字：纯词法模式下跨语言题会偏低；
且书籍要在"每本一文件夹"(course=书名)布局下，course 匹配才成立。

## 2026-08-09 · 首个黄金集真实基线（本地 e5-small，混检）

嵌入后端换成本地 `intfloat/multilingual-e5-small`（384 维，零 API、离线；因 bge-m3 太重压垮 Mac）。
在 Mac 上跑 `python -m kb_rag.eval --golden eval/golden.jsonl --k 8`，15 题：

| 分组 | R@1 | R@3 | R@5 | R@8 | MRR |
|---|---|---|---|---|---|
| **总体** | 0.60 | 0.73 | 0.80 | 0.87 | 0.683 |
| zh | 0.56 | 0.67 | 0.78 | 0.89 | 0.639 |
| en | 0.67 | 0.83 | 0.83 | 0.83 | 0.750 |
| book | 0.73 | 0.82 | 0.91 | **1.00** | 0.795 |
| course | 0.25 | 0.50 | 0.50 | 0.50 | 0.375 |

**结论**：
- 书籍类几乎不丢（R@8=1.00）——页码分节 + e5 配合好，这是主力语料。
- 两个 MISS：①「怎么给 agent 加审批」= e5-small 真实短板（中文问英文课程内容；同题英文版是 rank 1）；
  ②「multi-agent」原为黄金集写太严（命中 Agentic-Design-Patterns 第7章 Multi-Agent 本就合理，已把书加入可接受来源）。
- e5-small 的软肋是「中文→英文课程」这类跨语言 course 检索。可选提分：给 e5 加 query:/passage: 前缀、
  升 e5-base、或该类问题用 OpenAI 嵌入。个人自用当前水平已够。

## 检索精度杠杆：余弦阈值 + Rerank（可选，env 开关）

两个提精度的旋钮，默认关闭（0/off），按语料用黄金集调：

- **`KB_RAG_VEC_MIN_COS`**（余弦下限）：检索时丢弃余弦低于此值的向量命中，少喂无关块。
  先用 `kb_rag.scores` 看正确命中的 cos 一般多少，取个略低的值（如命中在 0.45+，可设 0.35）。设太高会误伤召回。
- **`KB_RAG_RERANK=1`**（交叉编码器精排）：混检先召回更大池（`KB_RAG_RERANK_POOL`，默认 30），
  再用交叉编码器把 (查询,文档) 一起打分、重排取 top-k。**对跨语言(中文问英文)提升最明显**，正补 e5-small 的短板。
  模型默认 `BAAI/bge-reranker-base`（中英，~1GB）；更强用 `KB_RAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3`。
  只在 top-N 上跑，不压垮机器；每查询多几百毫秒延迟。装：`pip install -e ".[local]"`（已含 sentence-transformers）。

**怎么量收益**——用黄金集对照跑（同一命令，加/不加 env）：
```
python -m kb_rag.eval --golden eval/golden.jsonl --k 8                    # 基线
KB_RAG_RERANK=1 python -m kb_rag.eval --golden eval/golden.jsonl --k 8    # 开精排
KB_RAG_VEC_MIN_COS=0.35 python -m kb_rag.eval --golden eval/golden.jsonl  # 加余弦阈值
```
对比 Recall@k / MRR，尤其看 zh（跨语言）那组。有提升就把 env 写进 ~/.zshrc 固化；没提升就别开（省延迟）。
固化后记得让问答用的 session 也带上同样的 env（否则检索和评测不一致）。

## 2026-08-14 · Rerank A/B（黄金集扩到 39 题，本地 e5-small + bge-reranker-base）

| 分组 | 基线 R@8 / MRR | +Rerank R@8 / MRR |
|---|---|---|
| ALL (n=39) | 0.87 / 0.771 | **0.90 / 0.782** |
| zh (n=24) | 0.79 / 0.649 | **0.83 / 0.667** |
| en (n=15) | 1.00 / 0.967 | 1.00 / 0.967 |
| book (n=21) | 1.00 / 0.893（R@1 0.86） | 1.00 / 0.873（R@1 0.81） |
| course (n=12) | 0.67 / 0.625 | 0.67 / 0.625 |
| video (n=6) | 0.83 / 0.639 | **1.00 / 0.778** |

**结论**：rerank 净正——video 大涨（R@8 0.83→1.00）、zh 跨语言小涨、ALL R@8 0.87→0.90，代价是 book R@1 微降 0.05（R@8 仍 1.00）+ 约几百 ms 延迟 + 1GB 模型。已建议固化 `KB_RAG_RERANK=1`。

**关键洞察**：剩下 4 个 MISS 全是 course（中文问英文课程），rerank 无能为力——因为正确来源**没进召回池**（top-30），这是**召回**盲区而非**排序**问题。rerank 只重排已召回的候选。根治需更强嵌入（e5-base）或 query 改写；但真实 agentic 回路多轮检索已能兜住（单发黄金集比实际严苛）。

## 2026-08-14 · 召回池诊断（RERANK_POOL 30 → 150）

为诊断 4 道顽固跨语言 course MISS 是"排序太深"还是"嵌入够不着"，撑大送进精排的候选数：

| 分组 | 默认(30) R@8/MRR | RERANK_POOL=150 R@8/MRR |
|---|---|---|
| ALL | 0.90 / 0.782 | **0.92 / 0.812** |
| zh | 0.83 / 0.667 | **0.88 / 0.736** |
| course | 0.67 / 0.625 | **0.75 / 0.625** |
| video | 1.00 / 0.778 | 1.00 / **1.000** |
| MISS | 4 | **3**（救回"什么是 RAG"） |

**结论**：两种问题都有。撑大召回池是**免费净赚**（不重嵌）——救回 1 道、zh 跨语言 R@8 0.83→0.88、
video MRR 满分，说明部分正确来源"捞得到只是排在 30 名之后"。**建议固化 `KB_RAG_RERANK_POOL=150`**
（代价：精排候选 30→150，每查询多约 1~2s，交互式可接受）。
剩余 3 道 course（审批/部署/多智能体分工）池撑到 600 仍捞不到 = e5-small 真实嵌入盲区，只有更强嵌入
（e5-base，768 维，需全量重嵌）能救——但这 3 道单发失败在真实 agentic 多轮回路里已能兜住，ROI 偏低。

## 2026-08-16 · 云端向量评测（OpenAI 256 维 + 无 rerank ＝ 线上真实配置）

WhatsApp 上云后，云端查询向量改用 **OpenAI `text-embedding-3-small`**，且压缩到 **256 维**
（`shrinkvecs`，见部署文档），云端容器**不装 torch → 没有 rerank**。此测复刻线上：
`KB_RAG_EMBEDDER=openai KB_RAG_OPENAI_DIM=256 KB_RAG_RERANK=0`，同一份 **39 题**黄金集，k=8。

| 分组 | R@1 | R@3 | R@5 | R@8 | MRR |
|---|---|---|---|---|---|
| **ALL** (n=39) | 0.67 | 0.87 | 0.87 | **0.95** | 0.777 |
| zh (n=24) | 0.67 | 0.83 | 0.83 | 0.92 | 0.762 |
| en (n=15) | 0.67 | 0.93 | 0.93 | 1.00 | 0.800 |
| book (n=21) | 0.76 | 0.90 | 0.90 | 0.95 | 0.840 |
| course (n=12) | 0.58 | 0.83 | 0.83 | 0.92 | 0.722 |
| video (n=6) | 0.50 | 0.83 | 0.83 | 1.00 | 0.663 |

**与本地 e5-small 对照（同 39 题，均无 rerank）——云端反而更强：**

| 分组 | e5-small R@8 / MRR | OpenAI-256 R@8 / MRR | 变化 |
|---|---|---|---|
| ALL | 0.87 / 0.771 | **0.95 / 0.777** | R@8 +0.08 |
| zh（跨语言） | 0.79 / 0.649 | **0.92 / 0.762** | 大涨 |
| **course**（老大难） | 0.67 / 0.625 | **0.92 / 0.722** | R@8 +0.25 |
| video | 0.83 / 0.639 | **1.00 / 0.663** | R@8 +0.17 |
| en | 1.00 / **0.967** | 1.00 / 0.800 | 同语言 top-1 略逊 |
| book | 1.00 / 0.893 | 0.95 / 0.840 | 微降 |

**结论**：
- **OpenAI 向量把"中文问英文课程"这个跨语言盲区基本补上了**——course R@8 0.67→**0.92**、zh R@8 0.79→**0.92**，
  连 e5 一直救不动的这块都上来了。这验证了选 B（云端 OpenAI 向量）**不只是为了能部署，检索质量本身也更好**。
- 代价：**英文同语言 top-1 略逊**（en MRR 0.967→0.800）——e5-small 对英文 rank-1 更锐；OpenAI-256 把英文正确源
  排到 top-3 内但不总是第 1。book 也微降。综合仍是 OpenAI 净胜（ALL R@8 0.87→0.95，MRR 持平）。
- 而且这还是**压到 256 维 + 无 rerank** 的结果；全维 1536 或加 rerank 只会更高——但云端够用了。
- **只剩 2 个 MISS**（e5 基线是 4 个）：①「禀赋效应是什么」(zh·book) ②「怎么给 agent 加人工审批」
  (zh→en·course，历史顽固题)。两道在真实 agentic 多轮回路里仍能兜住（单发黄金集比实际严苛）。

**一句话**：云端线上检索（OpenAI 256 维）比本地免费的 e5-small **整体更准、跨语言明显更强**，
只在英文 rank-1 上略让一分。个人自用完全够，且老大难的 course 跨语言问题实质改善。

> **别把上表读成「e5 不行」。** 上面的对照是**两边都关 rerank**（为复刻云端——云端装不了 torch，没有 rerank）。
> 但**本地 e5 平时是开 rerank 的**（`KB_RAG_RERANK=1` + `POOL=150` 已固化进 `~/.zshrc`），
> 本地实际水平是 **ALL R@8 0.92 / MRR 0.812**（见 8-14 池诊断），和云端 0.95 只差一点。
> 而且 e5 **免费、离线，英文 top-1 更锐**（en MRR 0.967 > OpenAI 的 0.800）。
> 结论：**两套各司其职、都不弱**——本地 e5（带 rerank）管你在编辑器里的免费问答；云端 OpenAI-256 管
> WhatsApp（云端跑不了本地模型，只能用 API 向量，顺带跨语言更强）。**本地不用换成 OpenAI。**

<!-- 下次评测从这里往下追加，保留上面的历史记录 -->
