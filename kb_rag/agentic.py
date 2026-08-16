"""
agentic.py — 真 agentic 回路：把 engine 的确定性工具注册成 LLM 的 tools，
让 LLM 自主驱动检索（原生 tool-use 循环），而不是写死管线。

对比 agent.py 的"半 agentic"（LLM 只在 4 个固定点做决策）：这里 LLM 自己决定
调哪个工具、用什么词搜、按 course/kind 缩范围、够不够、要不要 verify、何时给答案或拒答。

分工不变（BCA M08）：确定性工具在 engine（稳、可审计）；"调哪个/调几次/够没够"交给 LLM。

两种后端（用哪家 LLM 的 key 就用哪个）：
  ToolAgent        —— Anthropic Claude（messages + tools）
  OpenAIToolAgent  —— OpenAI（chat.completions + function calling）
共用：工具集、system 提示、确定性 veto 护栏。测试可注入假 client 模拟 tool-use 流。
"""
from __future__ import annotations

import json
import os

from .agent import Answer
from .retriever import Retriever

MAX_STEPS = int(os.environ.get("KB_RAG_MAX_STEPS", "12"))   # 回路步数上限（搜索/verify 用得快，给够）
CLIP = 600
# 护栏阈值：语义核对（embedding 余弦，抗意译/跨语言）优先；无 embedder 时退回词面核对。
SEM_FLOOR = float(os.environ.get("KB_RAG_SEM_FLOOR", "0.30"))    # 语义支撑下限（余弦）
VERIFY_FLOOR = float(os.environ.get("KB_RAG_VERIFY_FLOOR", "0.15"))  # 词面重合下限（退回用）

SYSTEM = """你是忠实的学习知识库助手，只能依据检索到的原始语料回答，严禁编造。

你有一组工具可反复调用来自主完成检索：
- search(query, course?, kind?, k?)：全库或按课程/类型(course/book/video)检索相关块。
- fetch_section(course, section)：整节按原文顺序取回（讲解/不漏细节）。
- outline(course) / list_courses()：了解库里有什么、某课有哪些节。
- verify(claim, chunk_ids)：核对某个说法是否被指定块支撑（防编造引用）。
- answer(answer, citation_ids, blocked, block_reason)：给出最终答案，必须调用它来结束。

工作方式（自主决定，不是固定流程）：
1) 先想清楚要答什么；必要时用 list_courses/outline 了解范围。
2) 用 search 检索，**先判断命中是否真能回答问题**：够了就往下走；**不够或不对，就换一批关键词再搜一轮**——
   试近义词 / 更具体的术语 / **换一种语言（中英互换，跨语言尤其要试）** / 或用 course/kind 缩范围。
   第一轮弱不要在弱材料上硬答，多搜一两轮往往能捞到对的来源（迭代式检索是本回路的关键）。
3) 跨领域问题：分别在相关课程/类型里检索后综合。
4) 起草答案前，对关键论断用 verify 核对（核对一两条关键点即可，别反复 verify 浪费步数）。
5) 材料**确实够回答**了就**尽快调 answer 收尾**，别无谓地反复搜/反复核对；但材料明显不足时先按第 2 步换关键词再搜一轮。调 answer 结束：
   - 有充分支撑 → answer 写连贯答案（只用检索到的材料），citation_ids 填支撑该答案的块 id，blocked=false。
   - 检索不到、或无支撑 → blocked=true，block_reason 说明为什么（例如"库外问题"），answer 留简短说明。
绝不使用检索材料之外的知识或臆测。宁可拒答，不可编造。"""


def _tools() -> list[dict]:
    return [
        {"name": "search",
         "description": "在知识库检索相关内容块(BM25+向量·RRF)。可用 course(课名子串)/kind 缩小范围。返回带 id/course/section 的 top-k 块。",
         "input_schema": {"type": "object", "properties": {
             "query": {"type": "string"},
             "course": {"type": "string", "description": "课名子串过滤，可选"},
             "kind": {"type": "string", "enum": ["course", "book", "video", "data"],
                      "description": "类型过滤，可选"},
             "k": {"type": "integer", "description": "返回数量，默认 8"}},
             "required": ["query"]}},
        {"name": "fetch_section",
         "description": "整节按原文顺序取回某课的某一节（教学/不漏细节）。",
         "input_schema": {"type": "object", "properties": {
             "course": {"type": "string"}, "section": {"type": "string"}},
             "required": ["course", "section"]}},
        {"name": "outline",
         "description": "列出某课的有序小节清单。",
         "input_schema": {"type": "object", "properties": {"course": {"type": "string"}},
                          "required": ["course"]}},
        {"name": "list_courses",
         "description": "列出库里有哪些课及块数。",
         "input_schema": {"type": "object", "properties": {}, "required": []}},
        {"name": "verify",
         "description": "确定性核对：claim 的实义词有多少被指定 chunk_ids 覆盖，防编造引用。",
         "input_schema": {"type": "object", "properties": {
             "claim": {"type": "string"},
             "chunk_ids": {"type": "array", "items": {"type": "string"}}},
             "required": ["claim", "chunk_ids"]}},
        {"name": "answer",
         "description": "给出最终答案并结束。只用检索到的材料；无支撑则 blocked=true。",
         "input_schema": {"type": "object", "properties": {
             "answer": {"type": "string"},
             "citation_ids": {"type": "array", "items": {"type": "string"}},
             "blocked": {"type": "boolean"},
             "block_reason": {"type": "string"}},
             "required": ["answer", "citation_ids", "blocked"]}},
    ]


def _openai_tools() -> list[dict]:
    """把中立工具集转成 OpenAI function-calling 格式。"""
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["input_schema"]}} for t in _tools()]


def _jloads(s: str | None) -> dict:
    try:
        return json.loads(s or "{}")
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}


class _BaseToolAgent:
    """共用：工具执行 + 确定性 veto 护栏。子类实现各家 LLM 的回路。"""
    def __init__(self, engine: Retriever, max_steps: int = MAX_STEPS):
        self.e = engine
        self.max_steps = max_steps

    def _compact(self, chunks: list[dict], meta: dict) -> list[dict]:
        out = []
        for c in chunks:
            meta[c["id"]] = {"course": c["course"], "section": c["section"],
                             "layer": c.get("layer", "raw")}
            out.append({"id": c["id"], "course": c["course"], "kind": c.get("kind"),
                        "section": c["section"], "order": c.get("order"),
                        "text": (c["text"] or "")[:CLIP]})
        return out

    def _dispatch(self, name: str, inp: dict, meta: dict) -> str:
        try:
            if name == "search":
                r = self.e.search(inp["query"], course=inp.get("course"),
                                  kind=inp.get("kind"), k=int(inp.get("k", 8)))
                return json.dumps(self._compact(r, meta), ensure_ascii=False)
            if name == "fetch_section":
                r = self.e.fetch_section(inp["course"], inp["section"])
                return json.dumps(self._compact(r, meta), ensure_ascii=False)
            if name == "outline":
                return json.dumps(self.e.outline(inp["course"]), ensure_ascii=False)
            if name == "list_courses":
                return json.dumps(self.e.list_courses(), ensure_ascii=False)
            if name == "verify":
                return json.dumps(self.e.verify(inp["claim"], inp.get("chunk_ids", [])),
                                  ensure_ascii=False)
            return json.dumps({"error": f"未知工具 {name}"}, ensure_ascii=False)
        except Exception as e:  # 工具错误回给 LLM，让它调整
            return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)

    def _finalize(self, ans: Answer, inp: dict, meta: dict) -> Answer:
        """把 answer 的最终裁决权交给代码，不信 LLM 自报的 blocked。

        护栏（zero 编造引用逃逸，不靠模型自觉）：
          1. LLM 自己说 blocked → 尊重（拒答）。
          2. 声称有答案但**没引用任何真实检索块** → 强制拒答。
          3. 有引用 → 用确定性 verify 核对答案与所引块的重合；过低 → 强制拒答。
        引用 id 只认本次会话检索见过的真实块（编造 id 直接丢弃）。
        """
        if inp.get("blocked"):
            ans.blocked = True
            ans.block_reason = inp.get("block_reason") or inp.get("answer") or "无支撑，拒答"
            return ans
        ans.answer = (inp.get("answer") or "").strip()
        valid = [cid for cid in inp.get("citation_ids", []) if cid in meta]
        ans.citations = [{"id": cid, **meta[cid]} for cid in valid]
        if not valid:
            ans.blocked = True
            ans.block_reason = "确定性护栏：答案未引用任何检索到的真实块（疑似编造），拒答"
            ans.trace.append("veto: 无有效引用 → 拦截")
            return ans
        # 语义核对优先（抗意译/跨语言）；检索器无 verify_semantic 时退回词面核对。
        checker = getattr(self.e, "verify_semantic", None)
        if checker is not None:
            score, floor, label = checker(ans.answer, valid).get("score", 0.0), SEM_FLOOR, "语义"
        else:
            score = self.e.verify(ans.answer, valid).get("coverage", 0.0)
            floor, label = VERIFY_FLOOR, "词面"
        ans.coverage = round(score, 3)
        ans.trace.append(f"veto: {label}核对 score={ans.coverage} floor={floor}")
        if score < floor:
            ans.blocked = True
            ans.block_reason = (f"确定性护栏：答案与所引块{label}支撑过低({ans.coverage}<{floor})，"
                                f"疑似无支撑，拒答")
        return ans


class ToolAgent(_BaseToolAgent):
    """Anthropic Claude 回路。client 可注入（测试用假 client）。"""
    def __init__(self, engine: Retriever, client=None, model: str | None = None,
                 max_steps: int = MAX_STEPS):
        super().__init__(engine, max_steps)
        self.model = model or os.environ.get("KB_RAG_MODEL", "claude-sonnet-5")
        if client is not None:
            self.client = client
        else:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("需要 `pip install anthropic`") from e
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("未设置 ANTHROPIC_API_KEY")
            self.client = anthropic.Anthropic()

    def ask(self, query: str) -> Answer:
        ans = Answer(query=query, mode="agentic")
        meta: dict = {}
        messages: list[dict] = [{"role": "user", "content": query}]
        tools = _tools()
        for step in range(1, self.max_steps + 1):
            # 最后一步强制收尾：必须调 answer（用已有材料作答/拒答，而不是空手超限）
            tc = {"type": "tool", "name": "answer"} if step == self.max_steps else {"type": "auto"}
            resp = self.client.messages.create(
                model=self.model, max_tokens=4096, system=SYSTEM,
                tools=tools, tool_choice=tc, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                ans.rounds = step
                ans.answer = "".join(getattr(b, "text", "") for b in resp.content
                                     if getattr(b, "type", None) == "text").strip()
                if not ans.answer:
                    ans.blocked, ans.block_reason = True, "模型未产出答案"
                return ans
            final = next((b for b in tool_uses if b.name == "answer"), None)
            if final is not None:
                ans.rounds = step
                return self._finalize(ans, final.input or {}, meta)
            results = []
            for b in tool_uses:
                out = self._dispatch(b.name, b.input or {}, meta)
                ans.trace.append(f"step{step}: {b.name}({json.dumps(b.input, ensure_ascii=False)[:80]})")
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            messages.append({"role": "user", "content": results})
        ans.rounds = self.max_steps
        ans.blocked, ans.block_reason = True, f"达到步数上限({self.max_steps})仍未给出答案"
        return ans


class OpenAIToolAgent(_BaseToolAgent):
    """OpenAI 回路（chat.completions + function calling）。只需 OPENAI_API_KEY。"""
    def __init__(self, engine: Retriever, client=None, model: str | None = None,
                 max_steps: int = MAX_STEPS):
        super().__init__(engine, max_steps)
        # 默认便宜档 gpt-4o-mini（自动服务/WhatsApp 每问几厘）。要更强改 KB_RAG_OPENAI_MODEL。
        self.model = model or os.environ.get("KB_RAG_OPENAI_MODEL", "gpt-4o-mini")
        if client is not None:
            self.client = client
        else:
            try:
                import openai
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("需要 `pip install openai`") from e
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("未设置 OPENAI_API_KEY")
            self.client = openai.OpenAI()

    def ask(self, query: str) -> Answer:
        ans = Answer(query=query, mode="agentic-openai")
        meta: dict = {}
        messages: list[dict] = [{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": query}]
        tools = _openai_tools()
        for step in range(1, self.max_steps + 1):
            # 最后一步强制收尾：必须调 answer
            tc = ({"type": "function", "function": {"name": "answer"}}
                  if step == self.max_steps else "auto")
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=tools, tool_choice=tc)
            msg = resp.choices[0].message
            calls = list(getattr(msg, "tool_calls", None) or [])
            am: dict = {"role": "assistant", "content": msg.content or ""}
            if calls:
                am["tool_calls"] = [{"id": c.id, "type": "function",
                                     "function": {"name": c.function.name,
                                                  "arguments": c.function.arguments}}
                                    for c in calls]
            messages.append(am)
            if not calls:
                ans.rounds = step
                ans.answer = (msg.content or "").strip()
                if not ans.answer:
                    ans.blocked, ans.block_reason = True, "模型未产出答案"
                return ans
            final = next((c for c in calls if c.function.name == "answer"), None)
            if final is not None:
                ans.rounds = step
                return self._finalize(ans, _jloads(final.function.arguments), meta)
            for c in calls:
                inp = _jloads(c.function.arguments)
                out = self._dispatch(c.function.name, inp, meta)
                ans.trace.append(f"step{step}: {c.function.name}({json.dumps(inp, ensure_ascii=False)[:80]})")
                messages.append({"role": "tool", "tool_call_id": c.id, "content": out})
        ans.rounds = self.max_steps
        ans.blocked, ans.block_reason = True, f"达到步数上限({self.max_steps})仍未给出答案"
        return ans
