# WhatsApp 前端上线纪实（本地 → Railway 云端）

> 记录 2026-08-15 把 Knowledge-RAG 的 WhatsApp 前端从"本地 Mac demo"做到"云端 24 小时常驻"的
> 全过程、每一个坑、根因和修法。给未来的自己（和接手的人）当排错手册。

---

## 0. 最终成果

一句话：**手机发问题 → Twilio → Railway 云端 → 本地混合检索(BM25+OpenAI 向量) → gpt-4o-mini 生成 →
带出处回发到手机**。关电脑也能答。花费：一次性建库嵌入 ~$1，之后每问几厘~一分。

- 后端：`kb_rag/server.py`（FastAPI + Twilio），跑在 Railway，固定域名。
- LLM：OpenAI `gpt-4o-mini`（便宜档，`KB_RAG_OPENAI_MODEL` 可改）。
- 检索：混检；云端查询向量用 OpenAI `text-embedding-3-small`（不装 torch，容器轻）。
- 向量库：本地全量嵌成 1536 维 → 压到 **256 维 float16**（`kb_rag/shrinkvecs.py`）好过上传上限。

部署这条线的完整操作手册见 [`deploy-railway.md`](./deploy-railway.md)。本文是**踩坑纪实**。

---

## 1. 架构：两个阶段

```
阶段一（上午·本地 demo）
  手机 → Twilio → cloudflared 临时隧道 → 你 Mac 上的 uvicorn → 检索+LLM → 回发
  痛点：电脑一关就没了；cloudflared URL 每次变

阶段二（下午·云端常驻）
  手机 → Twilio → Railway(固定域名) → 容器内 uvicorn → 检索+LLM → 回发
  数据(chunks.jsonl + 压缩向量)烤进 Docker 镜像，随 railway up 上传
```

关键取舍：**云端查询向量用 OpenAI 而不是本地 e5**——因为云端跑 torch 又重又容易 OOM。
代价是要把语料**用 OpenAI 重嵌一份**（本地 e5 那份 `vectors.npz` 原样保留、日常免费问答仍用它）。

---

## 2. 踩坑全记录（按发生顺序）

### 坑 1 — `/whatsapp` 一直返回 422（最难的一个）

- **现象**：手机消息到了服务器，`POST /whatsapp` 却回 `422 Unprocessable Entity`。
- **误判**：一开始以为是缺 `python-multipart`（Starlette 解析表单要它）。装上后**仍然 422**。
- **真因**：`server.py` 顶部有 `from __future__ import annotations`，把类型注解变成**字符串**。
  FastAPI 解析端点时，要把 `request: Request` / `background_tasks: BackgroundTasks` 解析成特殊注入类型，
  但这两个类是在 `create_app()` **函数内部局部 import** 的、模块全局里没有 → 解析失败 →
  FastAPI 把它们当成**必填查询参数** → 每条 webhook 都 422（报错体 `loc:["query","request"]`）。
- **修法**：删掉 `from __future__ import annotations`（本文件所有注解在 py3.10+ 运行期都安全）。
  容器里复现确认：带这行 → 422（报错体一模一样）；删掉 → 200。
- **附带**：`python-multipart` 确实也要装（并进了 `whatsapp` extra）；`/whatsapp` 加了 try/except
  兜底永远返回 200，避免 Twilio 见非 2xx 反复重投。
- **教训**：**FastAPI 端点 + `from __future__ import annotations` + 局部 import 特殊类型 = 坑**。
  422 报错体里的 `loc` 会告诉你 FastAPI 把参数当成了什么，是最快的线索。

### 坑 2 — 改了代码不生效

- **现象**：`sed` 删了那行，问题还在。
- **真因**：uvicorn 已经把旧代码加载进内存了。
- **修法**：改文件后**必须重启 uvicorn**。

### 坑 3 — 粘贴带 `#` 注释/中文把命令打断

- **现象**：`railway login` 报 `error: unexpected argument '#'`；`ls: 有文件: No such file or directory`。
- **真因**：把带 `#`/中文注释的整块粘进终端，shell 把注释内容当成了命令参数。
- **修法**：**只粘纯命令、一行一条、不带任何注释**。

### 坑 4 — railway 登录过期

- **现象**：`Unauthorized. Please run railway login again` / `invalid_grant`。
- **修法**：`railway login` 重新授权。

### 坑 5 — `railway up` 上传 413 Payload Too Large

- **现象**：`Failed to upload code. File too large (628275732 bytes)`。
- **真因**：把 727MB 的全维向量库直接塞进上传，超了 Railway CLI 的上传上限（有人 270MB 就被拒）。
- **修法（选了 B：本地压缩后再传）**：用 Matryoshka 特性把向量**截断降维 + 存 float16**：
  `kb_rag/shrinkvecs.py`，1536 → 384 → 256 维，727MB → ~69MB。检索质量几乎不掉
  （容器里验证：top-10 排序 fp16 vs fp32 截断 = 10/10 一致）。

### 坑 6 — 维度对不上

- **现象**：建索引打印 `(123156, 1536)`，不是想要的 512。
- **真因**：`export KB_RAG_OPENAI_DIM=512` 没落到实际跑命令的那个 shell。
- **教训**：**建索引的维度 = 云端 `KB_RAG_OPENAI_DIM` 必须一模一样**，否则查询向量和库对不上直接崩。
  后来统一走 shrinkvecs 的 `--dim`，云端变量跟着设同一个值。

### 坑 7 — 云端构建 `COPY chunks.jsonl` not found

- **现象**：上传过了，构建挂在 `[6/7] COPY chunks.jsonl`："/chunks.jsonl": not found。
- **真因**：Railway 上传时**同时套用 `.gitignore` 和 `.railwayignore`（取并集）**。
  `chunks.jsonl` 在 `.gitignore` 里被挡掉，没进上传包 → 构建时找不到。
- **修法**：把 `chunks.jsonl` 从 `.gitignore` 移出（它仍是本地产物，只是不再被上传排除）。

### 坑 8 — `railway up` operation timed out

- **现象**：`Compressed 100%` 后立刻 `Failed ... operation timed out`，连续两次。
- **真因**：上行带宽偏慢，~115MB 的包传不完就超时。
- **修法**：向量再压一档 **384 → 256 维**（包降到 ~85MB，和之前能传上去的 88MB 相当）。传上去了。
- **备选**：还超时就继续降到 128 维，或换更稳的网络。

### 坑 9 — healthcheck 失败：`$PORT is not a valid integer`

- **现象**：构建全过，健康检查却 `Error: Invalid value for '--port': '$PORT' is not a valid integer`。
- **真因**：`railway.json` 的 `startCommand` 里 `--port $PORT` 被 Railway **当字面字符串**执行，
  没经过 shell 展开，uvicorn 收到字面 `$PORT`。
- **修法**：把启动命令包进 shell：`sh -c 'uvicorn ... --port ${PORT:-8000}'`，`$PORT` 才会展开。

### 坑 10 — 收到了但回不了：Twilio 400 非法号码

- **现象**：`POST /whatsapp 200`（收到了、agent 也跑了），但手机没回。日志：
  `The 'From' number whatsapp:+14155238886⇥ is not a valid phone number`。
- **真因**：`TWILIO_WHATSAPP_FROM` 的值**末尾混进一个 tab**（粘贴带进去的），Twilio 判为非法号。
- **修法**：把变量值手动重敲干净；同时代码里给 `TWILIO_*` 读值加 `.strip()` 护栏，以后再混进空格也不怕。

### 坑 11 — GitHub 自动部署和 `railway up` 打架

- **现象**：一条 `via GitHub` 的部署 FAILED，用的是 `railpack` 不是我们的 Dockerfile。
- **真因**：Railway 服务连着 GitHub 仓库、会自动部署，但它从 **`main`** 构建——而我们的
  Dockerfile/railway.json/数据都在 `claude/structure-review-7uq4c1` 分支、没合进 main →
  找不到 Dockerfile → 退回 railpack → 又没数据 → 必失败。它没有替换掉 `railway up` 的好版本。
- **修法**：断开该服务的 GitHub Source（或关自动部署），只用 `railway up`（带 Dockerfile+数据）部署。
- **教训**：`railway up`（CLI 传本地镜像）和 GitHub 自动部署（从仓库构建）是**两条互斥的线**，
  数据在 gitignore 里进不了仓库时，只能走 `railway up`，就别让 GitHub 那条掺和。

---

## 3. 最终部署配置（定稿）

| 文件 | 作用 |
|------|------|
| `Dockerfile` | python:3.12-slim，只装 `.[whatsapp,vector]`（无 torch），烤进 `chunks.jsonl` + `vectors-openai-384.npz`，`sh -c` 展开 `$PORT` |
| `railway.json` | Dockerfile 构建；`startCommand` 用 `sh -c` 包住；`/health` 健康检查 |
| `.railwayignore` / `.dockerignore` | 放行语料+压缩向量，挡掉 727MB 全维库/e5 向量/`.git`/数据仓库 |
| `.gitignore` | **不再忽略 `chunks.jsonl`**（否则 railway up 并用 gitignore 会漏传） |
| `kb_rag/shrinkvecs.py` | 向量降维+半精度压缩（Matryoshka 截断 + fp16），好过上传上限 |
| `kb_rag/embed.py` | `OpenAIEmbedder` 支持 `KB_RAG_OPENAI_DIM` 降维 |
| `kb_rag/server.py` | 删 future-annotations；`/whatsapp` 兜底 200；`TWILIO_*` 读值 `.strip()` |

**Railway 必设的密钥变量**（不进镜像/git）：`OPENAI_API_KEY`、`TWILIO_ACCOUNT_SID`、
`TWILIO_AUTH_TOKEN`、`TWILIO_WHATSAPP_FROM`、`KB_RAG_SKIP_VALIDATION=1`、`KB_RAG_OPENAI_DIM=256`。

---

## 4. 花费

- **一次性**：全库用 OpenAI 嵌一遍 ≈ **$1**（12 万块）。不再重复；加新语料只嵌新增块（几分钱）。
- **每问一次**：查询 embedding ≈ $0.00001（可忽略）+ gpt-4o-mini 生成 ≈ **$0.002–0.01**。
- **看账单**：platform.openai.com → Settings → Usage，Group by **Model** 看拆分；设月度 spend limit 当护栏。
- 云端 Railway：按用量计费，小服务通常几美元/月。

---

## 5. 日常维护速查

- **更新云端数据**：本机 `./update.sh` → `buildindex --out vectors-openai.npz --embedder openai`
  补嵌新增块 → `shrinkvecs --dim 256` 重压 → `railway up`。（数据烤进镜像，必须 railway up 才更新。）
- **改模型**：Railway 改 `KB_RAG_OPENAI_MODEL`（`gpt-4o` 更强更贵），自动重启。
- **看在线状态**：浏览器开 `https://<域名>/health`，`{"ok":true}` = 活着。
- **看日志**：`railway logs`，发消息后应见 `POST /whatsapp ... 200`。

---

## 6. 一页纸教训

1. FastAPI 端点别配 `from __future__ import annotations` + 局部 import 特殊类型（→ 422）。
2. 改代码要重启进程；粘命令别带 `#`/中文注释。
3. `railway up` 有上传上限——大向量库先 `shrinkvecs` 降维+fp16 再传。
4. 建索引维度 = 云端 `KB_RAG_OPENAI_DIM`，必须一致。
5. Railway 上传**并用 gitignore + railwayignore**——需要上传的产物别被 gitignore 挡了。
6. 上传超时就把包压更小（降维），或换稳网络。
7. `railway.json` 的 `startCommand` 用 `sh -c` 包住才会展开 `$PORT`。
8. 环境变量读值加 `.strip()`——尾部空格/tab 会让 Twilio 判号码非法。
9. `railway up` 和 GitHub 自动部署互斥；数据进不了 git 时，断开 GitHub 只走 railway up。
10. 排错先看**日志/报错体**（422 的 `loc`、Twilio 的具体报错），一次定位，别盲试。
