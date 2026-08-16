"""
scores.py — 检索打分诊断。看每个查询的混检命中：rrf / cos(语义) / cov(词面覆盖) / 来源。

用来调参、排查召回：改 KB_RAG_LEX_MIN_COV / KB_RAG_W_VEC 等环境变量后重跑对比。

用法：
    python -m kb_rag.scores "向量检索 相似度" "智能体 推理 行动" "vector embedding"
    python -m kb_rag.scores --k 6 --chunks chunks.jsonl --vectors vectors.npz "问题1" "问题2"
    KB_RAG_LEX_MIN_COV=0.5 KB_RAG_W_VEC=2 python -m kb_rag.scores "中文问题"   # 带调参跑
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .embed import make_embedder
from .engine import Engine
from .hybrid import HybridRetriever


def main() -> None:
    ap = argparse.ArgumentParser(description="检索打分诊断（rrf/cos/cov）")
    ap.add_argument("queries", nargs="+", help="一个或多个查询")
    ap.add_argument("--chunks", type=Path, default=Path("chunks.jsonl"))
    ap.add_argument("--vectors", type=Path, default=Path("vectors.npz"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--course", default=None, help="限定课程（可选）")
    ap.add_argument("--kind", default=None, choices=["course", "book", "video", "data"])
    a = ap.parse_args()

    eng = Engine.from_jsonl(a.chunks)
    hy = HybridRetriever.from_store(eng, make_embedder(), a.vectors)
    print(f"调参: LEX_MIN_COV={os.environ.get('KB_RAG_LEX_MIN_COV', '0.0')} "
          f"VEC_MIN_COS={os.environ.get('KB_RAG_VEC_MIN_COS', '0.0')} "
          f"W_LEX={os.environ.get('KB_RAG_W_LEX', '1.0')} "
          f"W_VEC={os.environ.get('KB_RAG_W_VEC', '1.0')} "
          f"RERANK={os.environ.get('KB_RAG_RERANK', 'off')}  "
          f"embedder={type(hy.emb).__name__}")
    for q in a.queries:
        print(f"\nQ: {q}")
        hits = hy.search(q, course=a.course, kind=a.kind, k=a.k)
        if not hits:
            print("  （无命中）")
        for h in hits:
            print(f"  rrf={h['rrf']:.4f} cos={h['cos']:.3f} cov={h['coverage']:.2f} "
                  f"[{h['kind']}] {h['course'][:28]} · {h['section'][:26]}")


if __name__ == "__main__":
    main()
