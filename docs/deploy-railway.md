# 部署到 Railway（24 小时在线，关电脑也能用）

把 WhatsApp 后端从"本地 Mac"搬到 Railway，拿到**固定域名**、常驻在线。走**云端 OpenAI embedding**
路线：容器不装 torch，轻、启动快、不易 OOM。语料和向量库烤进镜像，随 `railway up` 从本机上传。

> 关键约束：**语料向量和查询向量必须用同一 embedder + 同一维度**。云端用 OpenAI
> `text-embedding-3-small`：先本机重嵌成全维 `vectors-openai.npz`（一次性，约 $1），再**压成 384 维**
> `vectors-openai-384.npz` 上云（好过 `railway up` 上传上限）；云端查询也用 384 维，两边对齐。
> 本地那份 e5 的 `vectors.npz` 原样保留、互不影响。

---

## 一次性：本机把语料重嵌成 OpenAI 向量

在 `~/Knowledge-RAG`（先确保 `.env` 里的 `OPENAI_API_KEY` 已填）：

```bash
cd ~/Knowledge-RAG
set -a; source .env; set +a          # 载入 OPENAI_API_KEY
# 先嵌成全维 1536（下一节再压到 384 上云）
python -m kb_rag.buildindex \
  --chunks chunks.jsonl \
  --out vectors-openai.npz \
  --embedder openai --batch 256
```

- 会打印进度、分批落盘；中断了再跑会**接着嵌**（已嵌的块复用，不重来）。
- 约 12 万块 ≈ 花 $1 出头、跑一阵（受 OpenAI TPM 限流影响，慢是正常的，会自动退避重试）。
- 跑完得到 `vectors-openai.npz`。**本地日常问答仍用免费的 e5**（`vectors.npz`），这份只给云端。

### 再压一步：降维 + 半精度，好过 railway up 上传上限

`vectors-openai.npz`（1536 维）约 727MB，`railway up` 收不下（会 413）。压成 384 维 float16（~95MB）：

```bash
python -m kb_rag.shrinkvecs --in vectors-openai.npz --out vectors-openai-384.npz --dim 384
```

- Matryoshka 截断，几秒、免费、无需重嵌；检索质量几乎不掉。
- 上云的是 `vectors-openai-384.npz`；原 `vectors-openai.npz` 留着备用。
- **云端 `KB_RAG_OPENAI_DIM` 必须 = 384**（和这里一致），否则维度对不上会崩。

---

### 如果 `railway up` 上传超时（operation timed out）

上行带宽慢时，~115MB 的包可能传不完就超时。把向量再压小一档即可（384→256 维，包降到 ~85MB）：

```bash
python -m kb_rag.shrinkvecs --in vectors-openai.npz --out vectors-openai-384.npz --dim 256
railway variables --set "KB_RAG_OPENAI_DIM=256"   # 云端维度跟着改成 256
railway up
```

还超时就再降到 `--dim 128`（同时 `KB_RAG_OPENAI_DIM=128`）。维度越低质量略降但先跑通为要。

## 装 Railway CLI 并登录

```bash
brew install railway            # 或： npm i -g @railway/cli
railway login                   # 浏览器登录你的 Railway 账号
```

## 建项目 + 首次部署

```bash
cd ~/Knowledge-RAG
railway init                    # 新建一个 Railway 项目（起个名，如 knowledge-rag-wa）
railway up                      # 上传构建上下文 → 按 Dockerfile 构建 → 部署
```

- `railway up` 会上传 `chunks.jsonl` + `vectors-openai-384.npz`（`.railwayignore` 已放行这两个、
  挡掉大的 727MB `vectors-openai.npz`/e5 向量/`.git`/RAG-Database）。首次上传 ~180MB，稍等。
- 构建按仓库根的 `Dockerfile` 走；`railway.json` 指定了启动命令(监听 `$PORT`)和 `/health` 健康检查。

## 配置环境变量（密钥只放这里，绝不进镜像/git）

面板 Variables 里加，或用 CLI：

```bash
railway variables \
  --set "OPENAI_API_KEY=sk-你的key" \
  --set "KB_RAG_OPENAI_MODEL=gpt-4o-mini" \
  --set "KB_RAG_OPENAI_DIM=384" \
  --set "KB_RAG_BACKEND=openai" \
  --set "KB_RAG_EMBEDDER=openai" \
  --set "TWILIO_ACCOUNT_SID=你的SID" \
  --set "TWILIO_AUTH_TOKEN=你的Token" \
  --set "TWILIO_WHATSAPP_FROM=whatsapp:+14155238886" \
  --set "KB_RAG_SKIP_VALIDATION=1"
```

> `KB_RAG_OPENAI_DIM` **必须** = 你 shrinkvecs 用的 `--dim`（384），否则维度对不上会报错。
> `KB_RAG_SKIP_VALIDATION=1` 先跳过 Twilio 验签，确保跑通；之后想收紧见文末。

## 拿公网域名

```bash
railway domain                  # 生成一个 https://xxxx.up.railway.app 固定域名
```

## 指到 Twilio

Twilio Sandbox 的 **"When a message comes in"** 填：

```
https://xxxx.up.railway.app/whatsapp      （Method: POST）
```

保存后用手机发消息测试。**这次电脑关了也照样回**——大脑在 Railway 上了。 🎉

---

## 日常运维

- **验证在线**：浏览器打开 `https://xxxx.up.railway.app/health`，看到 `{"ok":true,...}` 即活着。
- **看日志**：`railway logs`（或面板 Deployments → Logs）。发消息后应看到 `POST /whatsapp ... 200`。
- **改模型**：面板把 `KB_RAG_OPENAI_MODEL` 改成 `gpt-4o`（更强更贵）即可，改完自动重启。
- **加了新语料后更新云端**：本机 `./update.sh` 之后，再 `python -m kb_rag.buildindex
  --out vectors-openai.npz --embedder openai` 补嵌新增块 → 再 `python -m kb_rag.shrinkvecs`
  重新压出 `vectors-openai-384.npz` → 然后 `railway up` 重新部署。

## 花钱的地方

- Railway：按用量计费（小服务通常几美元/月）。
- OpenAI：每次回答 = 1 次查询 embedding（几乎免费）+ gpt-4o-mini 生成（每问几厘）。
- 一次性：语料重嵌 ~$1。

## 之后收紧安全（可选）

跑通后，想开启 Twilio 验签（防别人乱 POST 你的 webhook）：把 `KB_RAG_SKIP_VALIDATION` 删掉/设 0。
注意 Railway 在反代后面，`request.url` 可能是内网 http，需按转发头(`X-Forwarded-Proto/Host`)
重建公网 URL 再验签——这块要小改 `server.py`，需要时再说。
