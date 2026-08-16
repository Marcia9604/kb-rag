"""
rerank.py — 交叉编码器重排（cross-encoder rerank），检索精度的高性价比杠杆。

双编码器(embedding)把查询和文档**分开**编码再比余弦，快但粗；交叉编码器把 **(查询, 文档) 一起**
喂进模型、直接输出相关性分，准得多——尤其**跨语言**(中文问英文)，正好补 e5-small 的短板。
代价：每个候选都要过一次模型，所以**只对召回的 top-N 候选重排**(不是全库)，N 一般 20~30，很快。

流程：混检召回 top-N → 交叉编码器给每个 (query, doc) 打分 → 按新分排序取 top-k。

模型（本地、离线、免费）：
  默认 BAAI/bge-reranker-base（中英，约 1GB，较轻）
  更强跨语言用 bge-reranker-v2-m3（100+ 语言，约 2GB，更重）——`export KB_RAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3`
设备自动选 mps/cuda/cpu（复用 KB_RAG_LOCAL_DEVICE）。只在 top-N 上跑，即使重模型也压不垮机器。
"""
from __future__ import annotations

import os


class CrossEncoderReranker:
    def __init__(self, model: str | None = None):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("需要 `pip install sentence-transformers`（extra: local）") from e
        from .embed import _best_device
        self.model_name = model or os.environ.get("KB_RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
        self._m = CrossEncoder(self.model_name, device=_best_device())

    def rerank(self, query: str, rows: list[dict], top_k: int) -> list[dict]:
        """按交叉编码器分把 rows 重排，返回 top_k（每行加 rerank 分）。"""
        if not rows:
            return rows
        scores = self._m.predict([(query, r["text"]) for r in rows])
        order = sorted(range(len(rows)), key=lambda i: -float(scores[i]))
        out = []
        for i in order[:top_k]:
            r = dict(rows[i])
            r["rerank"] = round(float(scores[i]), 3)
            out.append(r)
        return out


_CACHE: dict = {}


def get_reranker(model: str | None = None) -> CrossEncoderReranker:
    """按模型名缓存，避免每次查询重新加载。"""
    key = model or os.environ.get("KB_RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
    if key not in _CACHE:
        _CACHE[key] = CrossEncoderReranker(model)
    return _CACHE[key]
