# Dockerfile — Knowledge-RAG WhatsApp 后端的云端镜像（Railway 用）。
# 走"云端 OpenAI embedding"路线：不装 torch/sentence-transformers，容器轻、启动快、不易 OOM。
# 语料(chunks.jsonl) + OpenAI 向量库(vectors-openai.npz) 直接烤进镜像，随 `railway up` 从本机上传。
FROM python:3.12-slim

WORKDIR /app

# 只装 whatsapp + vector 两组依赖（fastapi/uvicorn/twilio/python-multipart + numpy/openai）。
# 关键：**不装 local/torch**——云端查询向量用 OpenAI API 算，省内存。
COPY pyproject.toml ./
COPY kb_rag ./kb_rag
RUN pip install --no-cache-dir ".[whatsapp,vector]"

# 语料 + 向量库（gitignore 掉的私有大文件，靠 .railwayignore 放行上传）。
# 向量用**压缩版** vectors-openai-384.npz（1536→384 维 + float16，~95MB），好过 railway up 上传上限。
COPY chunks.jsonl ./chunks.jsonl
COPY vectors-openai-384.npz ./vectors-openai-384.npz

# 固定路径 + 云端后端。密钥(OPENAI/TWILIO)不写这里，走 Railway 环境变量注入。
ENV KB_RAG_CHUNKS=/app/chunks.jsonl \
    KB_RAG_VECTORS=/app/vectors-openai-384.npz \
    KB_RAG_BACKEND=openai \
    KB_RAG_EMBEDDER=openai \
    KB_RAG_OPENAI_DIM=256 \
    KB_RAG_OPENAI_MODEL=gpt-4o-mini \
    PORT=8000

# Railway 会注入 $PORT；uvicorn 必须监听它（railway.json 的 startCommand 也做了同样的事）。
CMD ["sh", "-c", "uvicorn kb_rag.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
