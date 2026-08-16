"""
retriever.py — 检索"接缝"。

agent（脑）只依赖这个接口，不关心底层怎么实现检索。规模变化时换的是实现，agent 不动：

  内存 BM25（现在，Engine）        —— 几千块以内，纯 stdlib、零依赖
  SqliteRetriever（~万级起）        —— SQLite FTS5(BM25) + 向量，单文件、支持增量
  VectorRetriever（百万级）         —— 独立向量库(pgvector/Qdrant) + 混合检索

只要实现下面五个方法、返回同样结构的 dict，就能整块替换。返回块统一形状：
  {id, course, kind, layer, section, order, text, ...(检索分数等可选)}
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Retriever(Protocol):
    """agent 可调的确定性工具集。任何检索后端实现它即可被 Agent 使用。"""

    def search(self, query: str, course: str | None = None,
               kind: str | None = None, k: int = 8) -> list[dict]:
        """相关块 top-k。course/kind 为过滤，None 表示全库。"""
        ...

    def fetch_section(self, course: str, section: str) -> list[dict]:
        """整节按原文顺序取回（教学模式：不漏细节）。"""
        ...

    def outline(self, course: str) -> list[str]:
        """该课的有序小节清单。"""
        ...

    def list_courses(self) -> dict:
        """{课名: 块数}，按块数降序。"""
        ...

    def verify(self, claim: str, chunk_ids: list[str]) -> dict:
        """确定性引用核对：claim 的实义词有多少被指定块覆盖。"""
        ...
