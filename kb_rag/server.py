"""
server.py — WhatsApp(Twilio)接入层。核心 Agent 不变，这里只做"消息 ↔ Agent"的适配。

分两层：
  1) Router（纯逻辑，可离线测）：把一条用户文本 → Agent 调用 → 渲染成 WhatsApp 文本块。
     维护 per-user 会话状态（记住选了哪门课），支持几个斜杠指令。
  2) Web 层（FastAPI）：Twilio webhook 收消息 → 验签 → 立即回 200 →
     后台线程跑 Agent（agentic 回路可能十几秒）→ 跑完用 Twilio REST 主动回发。
     不在 webhook 里同步等 Agent，避免 Twilio 15s 超时。

环境变量：
  KB_RAG_CHUNKS         chunks.jsonl 路径（默认 chunks.jsonl）
  KB_RAG_VECTORS        向量库 .npz 路径（默认 vectors.npz）；存在则自动用混检（跨语言必需）
  KB_RAG_BACKEND        openai | anthropic | mock（默认：有 OPENAI_API_KEY→openai，否则看 anthropic，再否则 mock）
  KB_RAG_OPENAI_MODEL   openai 后端模型，**默认 gpt-4o-mini（便宜，每问几厘）**；想更强改这个
  KB_RAG_MODEL          anthropic 后端模型（默认 claude-sonnet-5）
  OPENAI_API_KEY        openai 后端 + 查询嵌入用
  TWILIO_ACCOUNT_SID    Twilio SID
  TWILIO_AUTH_TOKEN     Twilio Auth Token（同时用于 webhook 验签）
  TWILIO_WHATSAPP_FROM  发送方号，如 "whatsapp:+14155238886"（沙箱号）
  KB_RAG_SKIP_VALIDATION 置 "1" 跳过 Twilio 签名校验（仅本地调试用）

跑（用你的 OpenAI key + 便宜模型）：
  pip install -e ".[whatsapp,vector]"     # fastapi + uvicorn + twilio + python-multipart + numpy + openai
  # ⚠️ python-multipart 必装：Starlette 解析 Twilio 表单靠它，缺了 /whatsapp 直接 422/500。
  #    已并进 whatsapp 这个 extra；单独补装：pip install python-multipart
  export OPENAI_API_KEY=sk-...
  export KB_RAG_OPENAI_MODEL=gpt-4o-mini  # 便宜档；这就是"怎么改模型"——改这个环境变量即可
  export TWILIO_ACCOUNT_SID=... TWILIO_AUTH_TOKEN=... TWILIO_WHATSAPP_FROM=whatsapp:+1...
  uvicorn kb_rag.server:app --host 0.0.0.0 --port 8000
  # 再用 ngrok 暴露 8000，把公网地址填进 Twilio Sandbox 的 webhook

改模型就一句话：`export KB_RAG_OPENAI_MODEL=<模型名>`（gpt-4o-mini 最便宜；gpt-4o 更强更贵）。

⚠️ 本文件**不要**加 `from __future__ import annotations`：FastAPI 端点 `/whatsapp` 的
   `request: Request` / `background_tasks: BackgroundTasks` 靠类型注解识别为特殊注入参数；
   开了 future-annotations 注解会变字符串，而 Request/BackgroundTasks 是在 create_app() 内部
   局部 import 的、模块全局里没有 → FastAPI 解析失败、把它们当查询参数 → 每条 webhook 都 422。
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .agent import Agent, Answer, AnthropicBackend, MockBackend
from .engine import Engine

WA_LIMIT = 1500          # 单条 WhatsApp 文本安全长度（Twilio 上限 ~1600）
HELP = (
    "📚 学习知识库助手\n"
    "· 直接发问题 → 全库检索问答\n"
    "· /courses → 看有哪些课\n"
    "· /use <课名关键词> → 选定一门课\n"
    "· /teach <主题> → 就当前课整节讲解（需先 /use）\n"
    "· /whoami → 看当前选的课\n"
    "· /help → 帮助"
)


# ---------------- 渲染 ----------------
def _bold(s: str) -> str:
    return f"*{s}*"


def render(ans: Answer) -> str:
    if ans.blocked:
        return f"🚫 无法回答：{ans.block_reason}"
    body = ans.answer or "(空)"
    parts = [body]
    if ans.citations:
        seen, cites = set(), []
        for c in ans.citations:
            key = (c["course"], c["section"])
            if key in seen:
                continue
            seen.add(key)
            cites.append(f"· 《{c['course']}》· {c['section']}")
        if cites:
            parts.append(_bold("来源") + "\n" + "\n".join(cites[:5]))
    return "\n\n".join(parts)


def split_chunks(text: str, limit: int = WA_LIMIT) -> list[str]:
    """按段落把长文本切成 <=limit 的多条消息，尽量不断句。"""
    if len(text) <= limit:
        return [text]
    out, buf = [], ""
    for para in text.split("\n\n"):
        piece = (buf + "\n\n" + para) if buf else para
        if len(piece) <= limit:
            buf = piece
            continue
        if buf:
            out.append(buf)
        # 单段仍超长 → 硬切
        while len(para) > limit:
            out.append(para[:limit])
            para = para[limit:]
        buf = para
    if buf:
        out.append(buf)
    return out


# ---------------- 会话 + 路由（纯逻辑，可离线测）----------------
@dataclass
class Session:
    course: str | None = None


@dataclass
class Router:
    agent: Agent
    sessions: dict[str, Session] = field(default_factory=dict)

    def _sess(self, user: str) -> Session:
        return self.sessions.setdefault(user, Session())

    def reply_for(self, user: str, text: str) -> list[str]:
        """一条用户消息 → 若干条回复文本（已切分）。不抛异常。"""
        try:
            return self._route(user, text)
        except Exception as e:  # 兜底：适配层任何异常都别把 webhook 打挂
            return [f"⚠️ 处理出错：{type(e).__name__}: {e}"]

    def _route(self, user: str, text: str) -> list[str]:
        s = self._sess(user)
        t = (text or "").strip()
        if not t:
            return [HELP]
        low = t.lower()

        if low in ("/help", "help", "帮助", "?", "？"):
            return [HELP]

        if low in ("/courses", "/course", "courses"):
            courses = self.agent.e.list_courses()
            if not courses:
                return ["（知识库为空）"]
            lines = [_bold("课程列表")]
            for c, n in courses.items():
                lines.append(f"· {c}（{n} 块）")
            return split_chunks("\n".join(lines))

        if low.startswith("/use"):
            q = t[4:].strip()
            if not q:
                return ["用法：/use <课名关键词>"]
            match = self._match_course(q)
            if not match:
                return [f"没找到匹配「{q}」的课。发 /courses 看有哪些。"]
            s.course = match
            return [f"✅ 已选定：《{match}》\n发 /teach <主题> 整节讲解，或直接发问题检索。"]

        if low.startswith("/whoami"):
            return [f"当前课：《{s.course}》" if s.course else "还没选课（/use <课名>）。"]

        if low.startswith("/teach"):
            topic = t[6:].strip()
            if not s.course:
                return ["先选课：/use <课名关键词>，再 /teach <主题>。"]
            if not topic:
                return ["用法：/teach <主题>"]
            if hasattr(self.agent, "_teach"):
                ans = self.agent._teach(topic, s.course)
            else:
                # 工具型 agent 没有独立 teach：用"完整讲解"的问法走普通回路（agent 会自己 outline+取节）
                ans = self.agent.ask(
                    f"请依据《{s.course}》完整、按原文顺序讲解「{topic}」，只用检索到的原文，不要发挥。")
            return split_chunks(render(ans))

        # 普通消息 → 搜索问答
        ans = self.agent.ask(t)
        return split_chunks(render(ans))

    def _match_course(self, q: str) -> str | None:
        courses = list(self.agent.e.list_courses().keys())
        ql = q.lower()
        exact = [c for c in courses if c.lower() == ql]
        if exact:
            return exact[0]
        sub = [c for c in courses if ql in c.lower()]
        return sub[0] if len(sub) >= 1 else None


# ---------------- 工厂：加载 Agent / Router ----------------
def build_router():
    chunks = Path(os.environ.get("KB_RAG_CHUNKS", "chunks.jsonl"))
    if not chunks.exists():
        raise RuntimeError(f"找不到语料 {chunks}；先跑 kb_rag.ingest 生成，或设 KB_RAG_CHUNKS")
    eng = Engine.from_jsonl(chunks)

    # 检索器：有 vectors.npz 就用混检（BM25+向量，跨语言必需），否则纯词法。
    vectors = Path(os.environ.get("KB_RAG_VECTORS", "vectors.npz"))
    retriever = eng
    if vectors.exists():
        try:
            from .embed import make_embedder
            from .hybrid import HybridRetriever
            retriever = HybridRetriever.from_store(eng, make_embedder(), vectors)
        except Exception as e:
            print(f"（向量库不可用，退回纯词法：{e}）")

    # 后端：openai（便宜自动服务，默认 gpt-4o-mini，走真 agentic tool-use 回路）
    #       / anthropic / mock。默认按环境 key 自动挑。
    backend = os.environ.get("KB_RAG_BACKEND") or (
        "openai" if os.environ.get("OPENAI_API_KEY") else
        "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "mock")
    if backend == "openai":
        from .agentic import OpenAIToolAgent
        agent = OpenAIToolAgent(retriever)          # 模型走 KB_RAG_OPENAI_MODEL，默认 gpt-4o-mini
    elif backend == "anthropic":
        agent = Agent(retriever, AnthropicBackend(model=os.environ.get("KB_RAG_MODEL")))
    else:
        agent = Agent(retriever, MockBackend())
    return Router(agent)


# ---------------- Web 层（FastAPI + Twilio）----------------
def create_app():
    """惰性构建 FastAPI app —— 只有真要起服务时才依赖 fastapi/twilio。"""
    from fastapi import BackgroundTasks, FastAPI, Request, Response
    from twilio.request_validator import RequestValidator
    from twilio.rest import Client

    router = build_router()
    # .strip()：环境变量粘贴时常混进尾部空格/tab，Twilio 会把带空白的号码判为非法 → 回发 400。
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    wa_from = os.environ.get("TWILIO_WHATSAPP_FROM", "").strip()
    skip_val = os.environ.get("KB_RAG_SKIP_VALIDATION", "").strip() == "1"
    twilio_client = Client(sid, token) if sid and token else None
    validator = RequestValidator(token) if token else None

    app = FastAPI(title="kb_rag WhatsApp")

    def _send(to: str, bodies: list[str]) -> None:
        if not (twilio_client and wa_from):
            return
        for b in bodies:
            twilio_client.messages.create(from_=wa_from, to=to, body=b)

    def _work(user: str, to: str, text: str) -> None:
        _send(to, router.reply_for(user, text))

    @app.get("/health")
    def health():
        return {"ok": True, "courses": len(router.agent.e.list_courses())}

    @app.post("/whatsapp")
    async def whatsapp(request: Request, background_tasks: BackgroundTasks):
        # 解析 Twilio 表单。form() 靠 python-multipart（whatsapp extra 已含）；
        # 万一缺库或负载异常，也**别抛 422/500**——Twilio 见非 2xx 会不断重投。兜底 200。
        try:
            form = await request.form()
            params = {k: str(v) for k, v in form.items()}
        except Exception as e:
            print(f"（/whatsapp 表单解析失败，忽略本条：{type(e).__name__}: {e}）")
            return Response(status_code=200)
        # Twilio 签名校验
        if validator is not None and not skip_val:
            sig = request.headers.get("X-Twilio-Signature", "")
            if not validator.validate(str(request.url), params, sig):
                return Response(status_code=403)
        sender = params.get("From", "")          # whatsapp:+E164
        body = params.get("Body", "")
        # 立即 200 签收，Agent 放后台跑完再回发
        background_tasks.add_task(_work, sender, sender, body)
        return Response(status_code=200)

    return app


# 便捷入口：uvicorn kb_rag.server:app
def __getattr__(name):
    if name == "app":
        return create_app()
    raise AttributeError(name)
