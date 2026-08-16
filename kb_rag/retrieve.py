"""
retrieve.py — 纯本地检索 CLI，**不调任何 LLM**。给"在本地 Claude Code 里让 Claude 当 agent"用。

省钱思路：检索(BM25+向量)是确定性的本地计算，几乎不花钱；真正贵的是推理那步(gpt-5.5 等)。
把推理交给你已经在付费的 Claude(本地 session)：Claude 调本命令拿到**带出处的原文+块 id**，
再自己按"只引用检索到的块、无支撑就拒答"的纪律作答——推理 0 API 费。唯一残留开销是 embedding
(text-embedding-3-small，极便宜，且只嵌新增)。

子命令：
    search  "问题" [--k 8] [--course 课名] [--kind book|course|video] [--full] [--json]
        混检(有 vectors.npz 就带语义，跨语言必需)。打印每条命中的 id / 来源 / 分数 / 原文。
    section "课/书名" "节名"        整节按原文顺序取回（如 section "思考，快与慢" "p.35-42"）
    outline "课/书名"              列出该课/书的所有节名（书就是 p.a-b 列表）
    courses                        列出全部课程/书/视频及块数
    verify  "答案" id1 id2 …        防幻觉硬护栏：答案 vs 所引块语义核对，不过就该拒答（退出码 2）

给 Claude 当 agent 的纪律（在本地 session 的 CLAUDE.md 已写明）：
    1) search 找证据 → 不够就换词/加 --course/--kind 再搜 → 必要时 section 取整节细节
    2) 只用检索到的原文作答，引用具体块 id；检索不到就明说"库里没有"，绝不臆测
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .engine import Engine

# 护栏阈值（与 agentic.py 一致）：语义核对优先，无向量时退回词面覆盖。
SEM_FLOOR = float(os.environ.get("KB_RAG_SEM_FLOOR", "0.30"))
VERIFY_FLOOR = float(os.environ.get("KB_RAG_VERIFY_FLOOR", "0.15"))


def _load(chunks: Path, vectors: Path | None):
    """有 vectors.npz → 混检(BM25+向量，跨语言必需)；否则纯词法 Engine。"""
    eng = Engine.from_jsonl(chunks)
    if vectors and vectors.exists():
        try:
            from .embed import make_embedder
            from .hybrid import HybridRetriever
            return HybridRetriever.from_store(eng, make_embedder(), vectors), True
        except Exception as e:  # 缺 numpy/openai key 等 → 退回词法，给出提示
            print(f"（提示：向量检索不可用，退回纯词法：{e}）", file=sys.stderr)
    return eng, False


def _clip(text: str, full: bool) -> str:
    text = " ".join(text.split())
    return text if full or len(text) <= 700 else text[:700] + " …[--full 看全文]"


def cmd_search(r, a) -> None:
    hits = r.search(a.query, course=a.course, kind=a.kind, k=a.k)
    if a.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return
    if not hits:
        print("（无命中。换个说法，或去掉 --course/--kind 再试。）")
        return
    for i, h in enumerate(hits, 1):
        cos = h.get("cos")
        score = f"cos={cos:.3f} " if cos is not None else ""
        print(f"\n[{i}] id={h['id']}  {score}cov={h.get('coverage',0):.2f}  "
              f"[{h['kind']}] 《{h['course']}》· {h['section']}")
        print(_clip(h["text"], a.full))


def cmd_section(r, a) -> None:
    rows = r.fetch_section(a.course, a.section)
    if not rows:
        print(f"（没找到：《{a.course}》· {a.section}。用 outline 看有哪些节名。）")
        return
    print(f"《{a.course}》· {a.section}  （{len(rows)} 块，按原文顺序）")
    for row in rows:
        print(f"\n— id={row['id']} —")
        print(" ".join(row["text"].split()))


def cmd_outline(r, a) -> None:
    secs = r.outline(a.course)
    if not secs:
        print(f"（没找到课/书：{a.course}。用 courses 看全部。）")
        return
    print(f"《{a.course}》共 {len(secs)} 节：")
    for s in secs:
        print(f"  · {s}")


def cmd_verify(r, a) -> None:
    """防幻觉硬护栏：答案 vs 所引块做语义核对；分数低于阈值 → 判定拒答（NOT SUPPORTED）。

    先做确定性检查：引用的 id 必须是真实存在、且是检索得到的块（防编造引用）。
    再做语义支撑：有向量用 verify_semantic(embedding 余弦，抗意译/跨语言)，否则退回词面覆盖。
    """
    ids = a.chunk_ids
    known = {c.id for c in (r.lex.chunks if hasattr(r, "lex") else r.chunks)}
    real = [i for i in ids if i in known]
    fake = [i for i in ids if i not in known]
    if not real:
        print(json.dumps({"verdict": "BLOCK", "reason": "未引用任何真实块（引用不存在）",
                          "fake_ids": fake}, ensure_ascii=False, indent=2))
        sys.exit(2)

    sem = getattr(r, "verify_semantic", None)
    if sem is not None:
        score = sem(a.claim, real)["score"]
        method, floor = "semantic(cos)", SEM_FLOOR
    else:
        score = r.verify(a.claim, real).get("coverage", 0.0)
        method, floor = "lexical(coverage)", VERIFY_FLOOR
    ok = score >= floor
    out = {"verdict": "SUPPORTED" if ok else "NOT_SUPPORTED",
           "method": method, "score": round(float(score), 3), "floor": floor,
           "checked_ids": real}
    if fake:
        out["warning_fake_ids"] = fake
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if not ok:
        print("→ 支撑不足：应改口拒答（知识库里没有足够依据），不要输出该答案。",
              file=sys.stderr)
        sys.exit(2)


def cmd_courses(r, a) -> None:
    cc = r.list_courses()
    print(f"共 {len(cc)} 个课程/书/视频：")
    for name, n in cc.items():
        print(f"  {n:6d}  {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="纯本地检索（不调 LLM），给本地 Claude 当 agent 用")
    ap.add_argument("--chunks", type=Path, default=Path("chunks.jsonl"))
    ap.add_argument("--vectors", type=Path, default=Path("vectors.npz"),
                    help="向量库(.npz)；有则混检(跨语言必需)，无则纯词法")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("search", help="混检，打印带出处的原文+块 id")
    ps.add_argument("query")
    ps.add_argument("--k", type=int, default=8)
    ps.add_argument("--course", default=None)
    ps.add_argument("--kind", default=None, choices=["course", "book", "video", "data"])
    ps.add_argument("--full", action="store_true", help="打印完整原文（默认截断 700 字）")
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(func=cmd_search)

    pt = sub.add_parser("section", help="整节按原文顺序取回")
    pt.add_argument("course")
    pt.add_argument("section")
    pt.set_defaults(func=cmd_section)

    po = sub.add_parser("outline", help="列出某课/书的所有节名")
    po.add_argument("course")
    po.set_defaults(func=cmd_outline)

    pv = sub.add_parser("verify", help="防幻觉护栏：答案 vs 所引块语义核对，不过就该拒答")
    pv.add_argument("claim", help="要核对的答案或关键论断")
    pv.add_argument("chunk_ids", nargs="+", help="该答案引用的块 id（search 结果里的 id=）")
    pv.set_defaults(func=cmd_verify)

    pc = sub.add_parser("courses", help="列出全部课程/书/视频")
    pc.set_defaults(func=cmd_courses)

    a = ap.parse_args()
    if not a.chunks.exists():
        print(f"✗ 找不到 {a.chunks}，先跑 kb_rag.ingest 生成 chunks.jsonl", file=sys.stderr)
        sys.exit(1)
    r, _ = _load(a.chunks, a.vectors)
    a.func(r, a)


if __name__ == "__main__":
    main()
