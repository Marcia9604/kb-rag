"""
agent.py — Agentic RAG 回路（吸取 （一个更早的项目） 教训①：要真 agentic，不是写死管线）。

核心 = 确定性工具(engine) + LLM 驱动的自主回路。分工照 BCA M08 黄金法则"能 workflow 就别 agent"：
  确定性 workflow（代码）：分块 / 检索 / RRF / 引用 verify —— 稳、可审计
  Agentic 自主（LLM）：① 规划 ② 迭代检索"够不够" ③ 综合 ④ 引用自查

回路：
  question
   → plan          拆子问题 / 选模式(search|teach) / 路由到课
   → retrieve-loop  调 search / fetch_section；judge 覆盖够不够 → 不够就重写再搜 ↺
   → synthesize     基于检索块起草，逐句挂引用
   → self-check     verify 每条引用；无支撑 → 补搜或拒答
   → Answer{answer, citations[], coverage, blocked}

后端可插（照 （一个更早的项目） spec 的 mock/可插做法）：
  MockBackend      —— 纯标准库、确定性，用启发式冒充 LLM 决策，离线即可测回路控制流
  AnthropicBackend —— 本地插 ANTHROPIC_API_KEY 激活，真 LLM 做规划/判断/综合/自查
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from .engine import Engine, tokenize
from .retriever import Retriever

MAX_ROUNDS = 3
COVERAGE_TARGET = 0.55       # 检索循环：top 命中覆盖达标即停
TEACH_SECTION_MIN = 0.7      # 教学模式：整节取回覆盖达标才算"不漏"


@dataclass
class Answer:
    query: str
    mode: str
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    coverage: float = 0.0
    rounds: int = 0
    blocked: bool = False
    block_reason: str = ""
    trace: list[str] = field(default_factory=list)


# ---------------- 后端接口 ----------------
class Backend:
    """agent 的"脑"。四个决策点，Mock 用启发式，Anthropic 用真 LLM。"""
    def plan(self, query: str, courses: list[str]) -> dict: ...
    def judge(self, query: str, hits: list[dict]) -> dict: ...
    def synthesize(self, query: str, hits: list[dict]) -> str: ...
    def selfcheck(self, answer: str, hits: list[dict]) -> list[str]: ...


class MockBackend(Backend):
    """确定性、纯 stdlib。冒充 LLM 决策，用来离线验证 agent 控制流是否正确。"""
    def plan(self, query, courses):
        # 简易路由：query 里点名了某门课就锁定它
        picked = next((c for c in courses if any(w and w.lower() in c.lower()
                       for w in re.split(r"\s+", query) if len(w) > 2)), None)
        return {"mode": "search", "subqueries": [query], "course": picked}

    def judge(self, query, hits):
        cov = hits[0]["coverage"] if hits else 0.0
        gap = cov < COVERAGE_TARGET
        # mock 的"重写"：加不了语义，就换成放宽过滤/加大 k（在 loop 里体现）
        return {"enough": not gap, "coverage": cov,
                "reformulation": query if gap else None}

    def synthesize(self, query, hits):
        if not hits:
            return ""
        lines = ["（Mock 抽取式综合——真 LLM 后端会在此基于以下原文生成连贯答案）"]
        for h in hits[:4]:
            snip = re.sub(r"\s+", " ", h["text"]).strip()[:180]
            lines.append(f"· {snip}… ⟨{h['course']}·{h['section']}⟩")
        return "\n".join(lines)

    def selfcheck(self, answer, hits):
        # mock：抽取式答案天然来自 hits，视为有支撑；无 hits 才算不支撑
        return [] if hits else ["整段答案无检索支撑"]


def _fmt_hits(hits: list[dict], limit: int = 8, clip: int = 700) -> str:
    """把检索块渲染成带编号的材料清单，喂给 Claude。"""
    lines = []
    for i, h in enumerate(hits[:limit], 1):
        text = re.sub(r"\s+", " ", h.get("text", "")).strip()[:clip]
        lines.append(f"[{i}] 《{h.get('course','?')}》· {h.get('section','?')}"
                     f" [{h.get('layer','?')}]\n{text}")
    return "\n\n".join(lines) if lines else "（无检索材料）"


class AnthropicBackend(Backend):
    """本地插 key 激活。lazy import，未装/无 key 时明确报错，不静默降级。

    四个决策点各发一条结构化 prompt 给 Claude（官方 anthropic SDK）：
      plan / judge / selfcheck → 结构化输出（output_config.format），低 effort
      synthesize               → 纯文本，只基于给定材料生成、无支撑则说明不足
    系统提示词固定在前（利于 prompt cache），query/材料等易变内容放 messages。
    模型默认 claude-sonnet-5（沿用原设定），可用 KB_RAG_MODEL 或构造参数覆盖。
    """
    def __init__(self, model: str | None = None):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("需要 `pip install anthropic` 才能用真后端") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("未设置 ANTHROPIC_API_KEY")
        import anthropic
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self._model = model or os.environ.get("KB_RAG_MODEL", "claude-sonnet-5")

    # ---------- 底层调用 ----------
    def _text_of(self, resp) -> str:
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def _call(self, system: str, user: str, *, schema: dict | None = None,
              max_tokens: int = 1024, effort: str = "low") -> str:
        """发一条请求；schema 非空则约束为该 JSON。返回文本（拒答/异常返回 ""）。"""
        output_config: dict = {"effort": effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_config=output_config,
            )
        except self._anthropic.APIError:
            return ""
        if resp.stop_reason == "refusal":
            return ""
        return self._text_of(resp)

    @staticmethod
    def _loads(text: str, default: dict) -> dict:
        try:
            return json.loads(text) if text else default
        except (json.JSONDecodeError, ValueError):
            return default

    # ---------- 四个决策点 ----------
    _PLAN_SYS = (
        "你是学习知识库助手的规划器。给定用户问题和课程清单，判断：\n"
        "① mode：需要整节有序讲解/教学 → teach；一般事实问答 → search。\n"
        "② course：若问题明确指向某门课，填该课名（须来自给定清单，可为原样子串）；否则 null。\n"
        "③ subqueries：把问题拆成 1~3 个检索子查询（保留关键术语）。\n"
        "只输出 JSON。"
    )
    _PLAN_SCHEMA = {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["search", "teach"]},
            "subqueries": {"type": "array", "items": {"type": "string"}},
            "course": {"type": ["string", "null"]},
        },
        "required": ["mode", "subqueries", "course"],
        "additionalProperties": False,
    }

    def plan(self, query, courses):
        clist = "、".join(courses) if courses else "（空）"
        text = self._call(
            self._PLAN_SYS,
            f"课程清单：{clist}\n用户问题：{query}",
            schema=self._PLAN_SCHEMA, max_tokens=512)
        d = self._loads(text, {})
        mode = d.get("mode") if d.get("mode") in ("search", "teach") else "search"
        subs = d.get("subqueries") or [query]
        course = d.get("course")
        if course and courses and not any(course.lower() in c.lower()
                                          or c.lower() in course.lower() for c in courses):
            course = None  # 幻觉课名，丢弃
        return {"mode": mode, "subqueries": subs, "course": course}

    _JUDGE_SYS = (
        "你是检索覆盖裁判。给定用户问题和已检索到的材料片段，判断这些材料是否"
        "足以准确、有支撑地回答问题。\n"
        "- enough：材料足够回答→true；有明显缺口→false。\n"
        "- coverage：0~1，材料对问题的覆盖程度。\n"
        "- reformulation：若不够，给一条改写后的检索查询以补齐缺口；够了则 null。\n"
        "只输出 JSON。"
    )
    _JUDGE_SCHEMA = {
        "type": "object",
        "properties": {
            "enough": {"type": "boolean"},
            "coverage": {"type": "number"},
            "reformulation": {"type": ["string", "null"]},
        },
        "required": ["enough", "coverage", "reformulation"],
        "additionalProperties": False,
    }

    def judge(self, query, hits):
        if not hits:
            return {"enough": False, "coverage": 0.0, "reformulation": query}
        text = self._call(
            self._JUDGE_SYS,
            f"用户问题：{query}\n\n检索材料：\n{_fmt_hits(hits)}",
            schema=self._JUDGE_SCHEMA, max_tokens=512)
        d = self._loads(text, {})
        # 兜底：解析失败时退回确定性覆盖信号，不阻断回路
        fallback_cov = hits[0].get("coverage", 0.0)
        enough = bool(d.get("enough", fallback_cov >= COVERAGE_TARGET))
        try:
            cov = float(d.get("coverage", fallback_cov))
        except (TypeError, ValueError):
            cov = fallback_cov
        reform = d.get("reformulation") if not enough else None
        return {"enough": enough, "coverage": round(cov, 2), "reformulation": reform}

    _SYNTH_SYS = (
        "你是忠实的学习助手。只依据下面给出的编号材料回答用户问题：\n"
        "- 严禁使用材料之外的知识或臆测；材料不足以回答时，明确说明缺什么、不要硬编。\n"
        "- 用清晰连贯的中文组织答案，可在关键处用 [n] 标注来源编号。\n"
        "- 若问题偏教学（整节讲解），按材料原有顺序有条理地展开，保证不漏关键细节。"
    )

    def synthesize(self, query, hits):
        if not hits:
            return ""
        return self._call(
            self._SYNTH_SYS,
            f"用户问题：{query}\n\n材料：\n{_fmt_hits(hits, limit=12, clip=1200)}",
            schema=None, max_tokens=4096, effort="medium")

    _CHECK_SYS = (
        "你是引用核查器。给定一段草稿答案和它所依据的编号材料，找出草稿中"
        "【没有任何材料支撑】的具体主张（编造、材料里查不到的内容）。\n"
        "- 只列真正无支撑的主张，逐条简述；能在材料中找到依据的不要列。\n"
        "- 全部有支撑则返回空数组。\n"
        "只输出 JSON。"
    )
    _CHECK_SCHEMA = {
        "type": "object",
        "properties": {"unsupported": {"type": "array", "items": {"type": "string"}}},
        "required": ["unsupported"],
        "additionalProperties": False,
    }

    def selfcheck(self, answer, hits):
        if not answer:
            return ["整段答案为空"]
        if not hits:
            return ["整段答案无检索支撑"]
        text = self._call(
            self._CHECK_SYS,
            f"草稿答案：\n{answer}\n\n所依据材料：\n{_fmt_hits(hits, limit=12, clip=1200)}",
            schema=self._CHECK_SCHEMA, max_tokens=1024)
        d = self._loads(text, {"unsupported": []})
        items = d.get("unsupported") or []
        return [str(x) for x in items if str(x).strip()]


# ---------------- Agent 回路 ----------------
class Agent:
    # 依赖 Retriever 接口而非具体 Engine —— 规模上来后换检索实现，agent 不改。
    def __init__(self, engine: Retriever, backend: Backend):
        self.e = engine
        self.b = backend

    def ask(self, query: str, k: int = 8) -> Answer:
        courses = list(self.e.list_courses().keys())
        plan = self.b.plan(query, courses)
        if plan.get("mode") == "teach" and plan.get("course"):
            return self._teach(query, plan["course"])
        return self._search(query, plan, k)

    def _search(self, query, plan, k):
        ans = Answer(query=query, mode="search")
        course = plan.get("course")
        ans.trace.append(f"plan: mode=search course={course or '全库'}")
        hits, q, rounds = [], query, 0
        while rounds < MAX_ROUNDS:
            rounds += 1
            hits = self.e.search(q, course=course, k=k)
            j = self.b.judge(query, hits)
            ans.trace.append(f"round{rounds}: 命中{len(hits)} 覆盖{j['coverage']:.2f} "
                             f"{'够' if j['enough'] else '不够→再搜'}")
            if j["enough"] or not hits:
                break
            # 迭代自纠：优先用 judge 给的改写查询"换关键词再搜一轮"（真·迭代检索）；
            # 没有改写时再退回放宽课程过滤 / 加大 k。
            reform = j.get("reformulation")
            if reform and reform.strip() and reform != q:
                q = reform
                ans.trace.append(f"  ↳ 换关键词重搜：{reform[:40]}")
            elif course:
                course = None
            else:
                k = min(k * 2, 24)
        ans.rounds = rounds
        ans.coverage = hits[0]["coverage"] if hits else 0.0
        if not hits:
            ans.blocked, ans.block_reason = True, "检索无命中——疑似库外问题，拒答"
            return ans
        draft = self.b.synthesize(query, hits)
        unsupported = self.b.selfcheck(draft, hits)
        if unsupported:
            ans.blocked = True
            ans.block_reason = "自查发现无引用支撑：" + "；".join(unsupported)
            return ans
        ans.answer = draft
        ans.citations = [{"course": h["course"], "section": h["section"],
                          "layer": h["layer"], "id": h["id"]} for h in hits[:5]]
        return ans

    def _teach(self, query, course):
        """教学模式：整节有序取回 + 对照大纲核查覆盖 → 保证不漏细节。"""
        ans = Answer(query=query, mode="teach")
        outline = self.e.outline(course)
        ans.trace.append(f"plan: mode=teach course={course} 小节{len(outline)}")
        # 选与 query 最相关的一节（mock：用 search 命中的 section；真后端可让 LLM 选）
        hit = self.e.search(query, course=course, k=1)
        section = hit[0]["section"] if hit else (outline[0] if outline else None)
        if not section:
            ans.blocked, ans.block_reason = True, f"{course} 无内容"
            return ans
        block = self.e.fetch_section(course, section)
        # 覆盖核查：整节 token 里被 query 触达的比例（简版；真后端对照大纲逐点核）
        sec_tokens = set(tokenize(" ".join(b["text"] for b in block)))
        q_tokens = set(tokenize(query))
        cov = (len(q_tokens & sec_tokens) / len(q_tokens)) if q_tokens else 0.0
        ans.coverage = round(cov, 2)
        ans.trace.append(f"fetch_section '{section}': {len(block)} 段有序，覆盖{cov:.2f}")
        ans.answer = self.b.synthesize(query, [{**b, "coverage": cov} for b in block])
        ans.citations = [{"course": course, "section": section,
                          "layer": b["layer"], "id": b["id"]} for b in block]
        if cov < TEACH_SECTION_MIN:
            ans.trace.append("⚠ 覆盖偏低：真 agent 会补搜相邻小节再教")
        return ans
