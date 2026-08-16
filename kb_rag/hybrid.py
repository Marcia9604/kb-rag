"""
hybrid.py — 真·混合检索：BM25 词法 + 向量语义，RRF 融合。实现 Retriever 接口。

对齐两份 RAG 文档的核心结论：**专名靠 BM25、意思靠向量，必须混**。之前 engine 的
"hybrid" 其实是 BM25+TFIDF 两路词法；这里补上语义那一半：

  lexical  = engine.search（BM25+TFIDF·RRF）            —— 专名/错误码/精确术语
  vector   = 查询向量 vs 块向量 余弦 top-n                —— 换说法/近义/概念
  融合      = RRF（只看名次，避免两边分数量纲不同）
  过滤      = course/kind 元数据先缩范围（大规模降噪关键）

确定性工具 fetch_section/outline/list_courses/verify 直接委托 engine（不需要向量）。
向量矩阵与 engine.chunks 顺序对齐；可存/取 .npy，避免每次重嵌。
"""
from __future__ import annotations

import os
from pathlib import Path

from .engine import Engine, tokenize

RRF_K = 60


class HybridRetriever:
    """实现 Retriever。lex=已建索引的 Engine；emb=Embedder；mat=(N,dim) 归一化向量。"""
    def __init__(self, engine: Engine, embedder, matrix):
        import numpy as np
        self.lex = engine
        self.emb = embedder
        self.mat = np.asarray(matrix, dtype="float32")
        if self.mat.shape[0] != len(engine.chunks):
            raise ValueError(f"向量数({self.mat.shape[0]})与块数({len(engine.chunks)})不一致")
        self._id2i = {c.id: i for i, c in enumerate(engine.chunks)}

    # ---------- 构建 / 持久化 ----------
    @staticmethod
    def _normalize(mat):
        import numpy as np
        n = np.linalg.norm(mat, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return (mat / n).astype("float32")

    @classmethod
    def build(cls, engine: Engine, embedder, batch: int = 256, log=print) -> "HybridRetriever":
        import numpy as np
        texts = [c.text for c in engine.chunks]
        vecs = []
        for i in range(0, len(texts), batch):
            vecs.extend(embedder.embed(texts[i:i + batch]))
            if log and (i // batch) % 10 == 0:
                log(f"  embedding {min(i + batch, len(texts))}/{len(texts)}")
        mat = cls._normalize(np.asarray(vecs, dtype="float32"))
        return cls(engine, embedder, mat)

    @classmethod
    def from_store(cls, engine: Engine, embedder, path: str | Path) -> "HybridRetriever":
        """从增量向量存储(.npz)按当前块顺序拼出对齐矩阵。"""
        from .vecstore import EmbeddingStore
        store = EmbeddingStore.load(path)
        mat = store.matrix_for([c.id for c in engine.chunks])
        return cls(engine, embedder, mat)

    # ---------- 内部 ----------
    def _allowed(self, course, kind):
        import numpy as np
        if not course and not kind:
            return None
        idx = [i for i, c in enumerate(self.lex.chunks)
               if (not course or course.lower() in c.course.lower())
               and (not kind or kind == c.kind)]
        return np.asarray(idx, dtype="int64")

    def _vector_rank(self, query, allow, n):
        import numpy as np
        q = np.asarray(self.emb.embed([query])[0], dtype="float32")
        q /= (np.linalg.norm(q) or 1.0)
        if allow is None:
            sims = self.mat @ q
            top = np.argsort(-sims)[:n]
            return [(int(i), float(sims[i])) for i in top]
        sub = self.mat[allow] @ q
        top = np.argsort(-sub)[:n]
        return [(int(allow[j]), float(sub[j])) for j in top]

    def _row(self, i, rrf, cos, qset):
        c = self.lex.chunks[i]
        tf = self.lex.tf[i]
        cov = (sum(1 for t in qset if t in tf) / len(qset)) if qset else 0.0
        return {"id": c.id, "course": c.course, "kind": c.kind, "section": c.section,
                "layer": c.layer, "order": c.order, "text": c.text,
                "rrf": round(rrf, 4), "cos": round(cos, 3), "coverage": round(cov, 2)}

    # ---------- Retriever 接口 ----------
    def search(self, query, course=None, kind=None, k=8):
        from collections import defaultdict
        # 可调（环境变量，便于按语料调参，不改代码）：
        #   KB_RAG_LEX_MIN_COV 词法命中覆盖低于此值就丢弃（防跨语言下的词法垃圾污染 RRF），默认 0=不过滤
        #   KB_RAG_VEC_MIN_COS 向量命中余弦低于此值就丢弃（检索时就降噪，少喂无关块），默认 0=不过滤
        #   KB_RAG_W_LEX / KB_RAG_W_VEC 词法/语义在 RRF 里的权重，默认各 1（加大语义可设 W_VEC=2）
        #   KB_RAG_RERANK=1 开交叉编码器精排（先召回更大池，精排后取 top-k；跨语言提升最明显）
        lex_min = float(os.environ.get("KB_RAG_LEX_MIN_COV", "0.0"))
        vec_min = float(os.environ.get("KB_RAG_VEC_MIN_COS", "0.0"))
        w_lex = float(os.environ.get("KB_RAG_W_LEX", "1.0"))
        w_vec = float(os.environ.get("KB_RAG_W_VEC", "1.0"))
        rerank_on = os.environ.get("KB_RAG_RERANK", "").lower() in ("1", "true", "yes", "on")
        rerank_pool = int(os.environ.get("KB_RAG_RERANK_POOL", "30"))
        take = max(k, rerank_pool) if rerank_on else k   # 开重排时先融合更大候选池
        # KB_RAG_POOL：召回池大小（词法/向量各取前 pool 个进融合）。默认 take*4；调大=把网撒大，
        # 用来诊断"正确来源到底捞不捞得到"——捞得到只是排后→调参能救；捞不到→才需要换更强的嵌入模型。
        pool = int(os.environ.get("KB_RAG_POOL") or max(take * 4, 20))
        lex = [h for h in self.lex.search(query, course=course, kind=kind, k=pool)
               if h["coverage"] >= lex_min]
        vec = self._vector_rank(query, self._allowed(course, kind), pool)
        fused = defaultdict(float)
        cos = {}
        for rank, h in enumerate(lex):
            fused[self._id2i[h["id"]]] += w_lex / (RRF_K + rank)
        for rank, (i, s) in enumerate(vec):
            if s < vec_min:
                continue
            fused[i] += w_vec / (RRF_K + rank)
            cos[i] = s
        qset = set(tokenize(query))
        out = sorted(fused.items(), key=lambda x: -x[1])[:take]
        rows = [self._row(i, rrf, cos.get(i, 0.0), qset) for i, rrf in out]
        if rerank_on and rows:
            from .rerank import get_reranker
            rows = get_reranker().rerank(query, rows, k)
        return rows

    def fetch_section(self, course, section):
        return self.lex.fetch_section(course, section)

    def outline(self, course):
        return self.lex.outline(course)

    def list_courses(self):
        return self.lex.list_courses()

    def verify(self, claim, chunk_ids):
        return self.lex.verify(claim, chunk_ids)

    def verify_semantic(self, claim: str, chunk_ids: list[str]) -> dict:
        """语义核对：答案 vs 所引块的 embedding 余弦（取最大）。抗意译/跨语言，供 veto 护栏用。"""
        import numpy as np
        ids = [c for c in chunk_ids if c in self._id2i]
        if not ids or not (claim or "").strip():
            return {"score": 0.0}
        q = np.asarray(self.emb.embed([claim])[0], dtype="float32")
        q /= (np.linalg.norm(q) or 1.0)
        sims = self.mat[[self._id2i[c] for c in ids]] @ q
        return {"score": float(sims.max())}
