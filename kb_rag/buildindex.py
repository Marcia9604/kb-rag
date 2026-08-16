"""
buildindex.py — 为 chunks.jsonl 生成/更新向量索引(.npz)，供 hybrid 混检用。**增量**：只嵌新块。

用法：
    pip install "kb-rag[vector]"        # numpy + openai
    export OPENAI_API_KEY=sk-...        # 或 VOYAGE_API_KEY / KB_RAG_EMBEDDER=fake
    python -m kb_rag.buildindex --chunks chunks.jsonl --out vectors.npz

扩充数据后再跑一次即可：已嵌过的块（id 相同）直接复用，只嵌新增块，不再全量重嵌。
换 embedding 模型后要**删掉 .npz 重跑**（维度/语义变了 → 全量重嵌）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .embed import make_embedder
from .engine import Engine
from .vecstore import EmbeddingStore


def main() -> None:
    ap = argparse.ArgumentParser(description="生成/增量更新向量索引供 hybrid 混检")
    ap.add_argument("--chunks", type=Path, default=Path("chunks.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("vectors.npz"))
    ap.add_argument("--embedder", default=None, help="openai|voyage|fake（默认看环境 key）")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()

    eng = Engine.from_jsonl(a.chunks)
    emb = make_embedder(a.embedder)
    store = EmbeddingStore.load(a.out) if a.out.exists() else EmbeddingStore.empty(emb.dim)

    missing = [c for c in eng.chunks if not store.has(c.id)]
    have = len(eng.chunks) - len(missing)
    print(f"块数 {len(eng.chunks)} | 已有向量 {have} | 需嵌 {len(missing)} "
          f"| embedder {type(emb).__name__} dim={emb.dim}")
    if not missing:
        print("✅ 全部已嵌，无需更新")
        return

    for i in range(0, len(missing), a.batch):
        part = missing[i:i + a.batch]
        store.add([c.id for c in part], emb.embed([c.text for c in part]))
        store.save(a.out)   # 分批落盘，中断也不白跑
        print(f"  已嵌 {min(i + a.batch, len(missing))}/{len(missing)}", flush=True)
    print(f"✅ 向量索引 {store.mat.shape} → {a.out}（新增 {len(missing)} 块）")


if __name__ == "__main__":
    main()
