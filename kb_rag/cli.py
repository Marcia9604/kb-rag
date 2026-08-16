"""cli.py — 命令行入口。

    python -m kb_rag.ingest --src knowledge_base/raw --out chunks.jsonl   # 建语料
    python -m kb_rag.cli ask   "temperature 怎么调"                        # 搜索 agent
    python -m kb_rag.cli teach "监督学习" --course "Sample"                # 教学 agent
    python -m kb_rag.cli courses

后端默认 mock（纯 stdlib）。加 --backend anthropic 且本地有 ANTHROPIC_API_KEY 时用真 LLM。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .agent import Agent, MockBackend, AnthropicBackend, Answer
from .engine import Engine


def _mk_agent(chunks: Path, backend: str, model: str | None = None) -> Agent:
    eng = Engine.from_jsonl(chunks)
    be = AnthropicBackend(model=model) if backend == "anthropic" else MockBackend()
    return Agent(eng, be)


def _render(a: Answer) -> str:
    L = [f"❓ {a.query}   ·  模式={a.mode}  轮次={a.rounds or '-'}  覆盖={a.coverage}",
         "─" * 66]
    for t in a.trace:
        L.append(f"  · {t}")
    L.append("─" * 66)
    if a.blocked:
        L.append(f"🚫 拒答：{a.block_reason}")
        return "\n".join(L)
    L.append(a.answer or "(空)")
    if a.citations:
        L.append("\n引用：")
        for c in a.citations[:6]:
            L.append(f"  ↪ 《{c['course']}》· {c['section']} [{c['layer']}]")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(prog="kb_rag")
    ap.add_argument("--chunks", type=Path, default=Path("chunks.jsonl"))
    ap.add_argument("--backend", choices=["mock", "anthropic"], default="mock")
    ap.add_argument("--model", default=None,
                    help="anthropic 后端模型（默认 claude-sonnet-5，可用 KB_RAG_MODEL 覆盖）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("ask", help="搜索 agent")
    pa.add_argument("q")
    pa.add_argument("--k", type=int, default=8)
    pa.add_argument("--agentic", action="store_true",
                    help="用真 agentic 回路（LLM 自主 tool-use）")
    pa.add_argument("--llm", choices=["openai", "anthropic"], default=None,
                    help="agentic 用哪家 LLM（默认看环境 key：有 OPENAI 用 openai，有 ANTHROPIC 用 anthropic）")
    pa.add_argument("--hybrid", action="store_true",
                    help="用混合检索（BM25+向量·RRF，需先 buildindex 生成 --vectors）")
    pa.add_argument("--vectors", type=Path, default=Path("vectors.npz"),
                    help="向量索引路径（配 --hybrid，buildindex 生成的 .npz）")
    pt = sub.add_parser("teach", help="教学 agent（整节有序取回）")
    pt.add_argument("q")
    pt.add_argument("--course", required=True)
    sub.add_parser("courses", help="列课程")
    a = ap.parse_args()

    ag = _mk_agent(a.chunks, a.backend, a.model)
    if a.cmd == "courses":
        for c, n in ag.e.list_courses().items():
            print(f"  {n:5d}  {c}")
    elif a.cmd == "ask":
        retriever = ag.e
        if a.hybrid:
            from .embed import make_embedder
            from .hybrid import HybridRetriever
            retriever = HybridRetriever.from_store(ag.e, make_embedder(), a.vectors)
        if a.agentic:
            llm = a.llm or ("openai" if os.environ.get("OPENAI_API_KEY")
                            and not os.environ.get("ANTHROPIC_API_KEY") else "anthropic")
            if llm == "openai":
                from .agentic import OpenAIToolAgent
                agent = OpenAIToolAgent(retriever, model=a.model)
            else:
                from .agentic import ToolAgent
                agent = ToolAgent(retriever, model=a.model)
            print(_render(agent.ask(a.q)))
        elif a.hybrid:
            from .agent import Agent
            print(_render(Agent(retriever, ag.b).ask(a.q, k=a.k)))
        else:
            print(_render(ag.ask(a.q, k=a.k)))
    elif a.cmd == "teach":
        # 强制 teach 模式：直接走教学路径
        print(_render(ag._teach(a.q, a.course)))


if __name__ == "__main__":
    main()
