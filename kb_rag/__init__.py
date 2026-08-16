"""kb_rag — 面向个人学习知识库的 agentic RAG（占位名，可改）。

- engine : 确定性检索核 + 工具层（纯 stdlib）
- ingest : 原始资料 → chunks.jsonl（layer=raw）
- agent  : agentic 回路（Mock 后端 stdlib 可测；AnthropicBackend 本地插 key 激活）
"""
from .engine import Engine, Chunk          # noqa: F401
from .agent import Agent, MockBackend, AnthropicBackend, Answer  # noqa: F401

__all__ = ["Engine", "Chunk", "Agent", "MockBackend", "AnthropicBackend", "Answer"]
