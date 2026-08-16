"""
eval.py — 黄金集评估：把"感觉还行"变成可重复、可报的数字。

评的是**检索层**(确定性、免费、可重复)：给一批 (问题, 应命中来源)，算正确来源有没有进 top-k。
这是 RAG 表现最关键、也最便宜修的一层。生成层(答案正确率/幻觉率)需要真跑 LLM，另说。

黄金集格式(每行一个 JSON，见 eval/golden.jsonl)：
    {"q": "什么是锚定效应？", "lang": "zh",
     "expect": {"kind": "book", "section_contains": "锚定"}}
    {"q": "How does HITL approval work?", "lang": "en",
     "expect": {"course": "Building AI Research Agents", "section_contains": "Human-in-the-Loop"}}
  - expect 可以是单个 dict，或多个可接受来源的 list（命中任一即算对）。
  - 匹配规则：hit 满足 expect 里所有给定字段才算命中：
      kind 相等 / course 子串命中 / section_contains 是 section 的子串。

指标：
    Recall@k   —— 有多少比例的问题，正确来源进了 top-k（越高越好）
    Recall@1/3/5、MRR —— 更细的排名质量
  另按 语言(zh/en) 和 类型(book/course/video) 分组，单独看跨语言检索健康度。

用法：
    python -m kb_rag.eval --golden eval/golden.jsonl --k 8
    python -m kb_rag.eval --golden eval/golden.jsonl --verbose      # 列出每题命中排名
无 vectors.npz 时自动退回纯词法（跨语言会偏低，仅供快速自检）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import Engine


def _load(chunks: Path, vectors: Path):
    eng = Engine.from_jsonl(chunks)
    if vectors and vectors.exists():
        try:
            from .embed import make_embedder
            from .hybrid import HybridRetriever
            return HybridRetriever.from_store(eng, make_embedder(), vectors), "hybrid(混检)"
        except Exception as e:
            print(f"（向量检索不可用，退回纯词法：{e}）", file=sys.stderr)
    return eng, "lexical(纯词法)"


def _norm_expect(expect) -> list[dict]:
    return expect if isinstance(expect, list) else [expect]


def _entry_kind(item: dict) -> str:
    """黄金集条目的期望类型（取第一个 expect 的 kind），用于分组统计。"""
    return _norm_expect(item["expect"])[0].get("kind", "other")


def _hit_matches(hit: dict, exp: dict) -> bool:
    if "kind" in exp and hit.get("kind") != exp["kind"]:
        return False
    if "course" in exp and exp["course"].lower() not in (hit.get("course") or "").lower():
        return False
    if "section_contains" in exp and exp["section_contains"] not in (hit.get("section") or ""):
        return False
    return "course" in exp or "section_contains" in exp  # 至少给一个定位字段


def _rank_of_first_match(hits: list[dict], expects: list[dict]) -> int:
    """返回第一条命中任一 expect 的名次(1-indexed)；top-k 内无命中返回 0。"""
    for i, h in enumerate(hits, 1):
        if any(_hit_matches(h, e) for e in expects):
            return i
    return 0


def evaluate(retriever, golden: list[dict], k: int, verbose: bool) -> dict:
    ranks = []
    for item in golden:
        hits = retriever.search(item["q"], k=k)
        r = _rank_of_first_match(hits, _norm_expect(item["expect"]))
        ranks.append(r)
        if verbose:
            top = hits[0] if hits else {}
            mark = f"✓ rank {r}" if r else "✗ MISS"
            src = f"[{top.get('kind','?')}] {top.get('course','')[:20]}·{top.get('section','')[:22]}"
            print(f"  {mark:10} {item.get('lang','?')} {item['q'][:34]:36} top1={src}")
    return {"ranks": ranks}


def _metrics(ranks: list[int], k: int) -> dict:
    n = len(ranks) or 1
    def recall_at(kk): return sum(1 for r in ranks if 1 <= r <= kk) / n
    mrr = sum((1.0 / r) for r in ranks if r) / n
    return {"n": len(ranks), "recall@1": recall_at(1), "recall@3": recall_at(3),
            "recall@5": recall_at(5), f"recall@{k}": recall_at(k), "mrr": mrr}


def _print_metrics(title: str, ranks: list[int], k: int) -> None:
    if not ranks:
        return
    m = _metrics(ranks, k)
    print(f"  {title:14} n={m['n']:3d}  R@1={m['recall@1']:.2f} R@3={m['recall@3']:.2f} "
          f"R@5={m['recall@5']:.2f} R@{k}={m[f'recall@{k}']:.2f}  MRR={m['mrr']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="黄金集检索评估（Recall@k / MRR）")
    ap.add_argument("--golden", type=Path, default=Path("eval/golden.jsonl"))
    ap.add_argument("--chunks", type=Path, default=Path("chunks.jsonl"))
    ap.add_argument("--vectors", type=Path, default=Path("vectors.npz"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--verbose", action="store_true", help="列出每题命中排名")
    a = ap.parse_args()

    if not a.golden.exists():
        print(f"✗ 找不到黄金集 {a.golden}", file=sys.stderr)
        sys.exit(1)
    golden = [json.loads(l) for l in a.golden.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not a.chunks.exists():
        print(f"✗ 找不到 {a.chunks}，先跑 kb_rag.ingest", file=sys.stderr)
        sys.exit(1)

    retriever, mode = _load(a.chunks, a.vectors)
    print(f"检索模式：{mode} | 黄金集 {len(golden)} 题 | k={a.k}\n")

    res = evaluate(retriever, golden, a.k, a.verbose)
    ranks = res["ranks"]

    print("\n总体：")
    _print_metrics("ALL", ranks, a.k)
    print("\n按语言：")
    for lang in ("zh", "en"):
        _print_metrics(lang, [r for r, it in zip(ranks, golden) if it.get("lang") == lang], a.k)
    print("\n按类型：")
    for kind in ("book", "course", "video", "other"):
        sub = [r for r, it in zip(ranks, golden) if _entry_kind(it) == kind]
        _print_metrics(kind, sub, a.k)

    misses = [it["q"] for r, it in zip(ranks, golden) if not r]
    if misses:
        print(f"\n未命中（正确来源没进 top-{a.k}，最该修的检索问题）：")
        for q in misses:
            print(f"  ✗ {q}")


if __name__ == "__main__":
    main()
