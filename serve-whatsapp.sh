#!/usr/bin/env bash
# serve-whatsapp.sh — 一键起 WhatsApp 前端：加载 .env → 起 uvicorn → 起 cloudflared 隧道。
#
#   1) 先复制模板并填好 key：   cp .env.example .env   然后编辑 .env
#   2) 一条命令启动：           ./serve-whatsapp.sh
#   3) 脚本会打印一条公网 URL，把它 + "/whatsapp" 填进 Twilio Sandbox 的 webhook：
#        https://xxxx.trycloudflare.com/whatsapp   (Method: POST)
#   4) 用手机发消息即可；按 Ctrl-C 一次性关掉 uvicorn 和 cloudflared。
#
# 说明：用的是 cloudflared 的免费临时隧道（quick tunnel），每次 URL 会变，
#       所以每次启动都要把新 URL 贴回 Twilio。要固定 URL 需正式域名（以后再说）。
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"

# --- 1) 加载 .env（key 只在这里，绝不入库）---
if [[ ! -f .env ]]; then
  echo "❌ 没找到 .env。先跑： cp .env.example .env  然后把 key 填进去。"
  exit 1
fi
set -a; source .env; set +a
if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == sk-填* ]]; then
  echo "❌ .env 里的 OPENAI_API_KEY 还没填。"
  exit 1
fi

# --- 2) 起 uvicorn（后台）---
echo "▶ 启动 uvicorn :${PORT} …（首次会加载本地 e5 向量模型，稍等几秒）"
uvicorn kb_rag.server:app --host 0.0.0.0 --port "${PORT}" &
UVICORN_PID=$!

# 关闭时一起清理
cleanup() {
  echo ""
  echo "⏹ 关闭 …"
  kill "${UVICORN_PID}" 2>/dev/null || true
  [[ -n "${CF_PID:-}" ]] && kill "${CF_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 等 uvicorn 起来（探 /health）
for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "✅ uvicorn 就绪"
    break
  fi
  sleep 1
  if [[ $i -eq 30 ]]; then echo "❌ uvicorn 30s 内没起来，看上面日志。"; exit 1; fi
done

# --- 3) 起 cloudflared 临时隧道，抓出公网 URL ---
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "❌ 没装 cloudflared。装： brew install cloudflared"
  exit 1
fi
echo "▶ 启动 cloudflared 隧道 …"
CF_LOG="$(mktemp)"
cloudflared tunnel --url "http://127.0.0.1:${PORT}" >"${CF_LOG}" 2>&1 &
CF_PID=$!

# 从日志里等出 trycloudflare URL
URL=""
for i in $(seq 1 30); do
  URL="$(grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' "${CF_LOG}" | head -1 || true)"
  [[ -n "${URL}" ]] && break
  sleep 1
done

echo ""
echo "══════════════════════════════════════════════════════════════"
if [[ -n "${URL}" ]]; then
  echo "✅ 全部就绪。把下面这个 webhook 贴进 Twilio Sandbox（POST）："
  echo ""
  echo "      ${URL}/whatsapp"
  echo ""
  echo "   然后用手机给沙箱号发消息即可。"
else
  echo "⚠ 没抓到 trycloudflare URL，隧道日志在： ${CF_LOG}"
fi
echo "   Ctrl-C 一次性关闭 uvicorn + cloudflared。"
echo "══════════════════════════════════════════════════════════════"

# 前台等着，Ctrl-C 触发 cleanup
wait "${CF_PID}"
