"""
embed.py — 可插拔的 embedding 后端。语义检索用它把文本转向量。

接口 Embedder：dim + embed(texts)->向量列表。换供应商只换实现，其余不动。
  LocalEmbedder   —— 本地模型(bge-m3，1024维)，**零 API、离线、免费**，跨语言强，适合个人自用
  OpenAIEmbedder  —— API，text-embedding-3-small(1536维)，算力外包，适合低配机器/大规模
  VoyageEmbedder  —— API，voyage-3，多语言(中文)更强
  FakeEmbedder    —— 确定性、纯 stdlib，用 token 袋哈希成向量，离线测融合机制用（无语义质量）

⚠️ 换 embedding 模型 = 维度/语义变了 → 必须**全量重嵌 + 重建索引**（两份 RAG 文档都强调）。
"""
from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol

from .engine import tokenize


class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> list[list[float]]:
        """一批文本 → 一批向量（未必归一化，检索侧统一归一化）。"""
        ...


class FakeEmbedder:
    """确定性假向量：token 袋哈希进固定桶。共享 token 多的文本向量更近——够验证融合，无真语义。"""
    def __init__(self, dim: int = 64):
        self.dim = dim

    def _one(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in tokenize(text) or [text[:8]]:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


def _subbatches(texts, char_budget: int, max_items: int):
    """按字符预算(近似 token)+ 条数上限切小批，避免单次请求超 TPM。超长单条截断。"""
    cur, n = [], 0
    for t in texts:
        t = (t or " ")[:char_budget]
        if cur and (n + len(t) > char_budget or len(cur) >= max_items):
            yield cur
            cur, n = [], 0
        cur.append(t)
        n += len(t)
    if cur:
        yield cur


class OpenAIEmbedder:
    """OpenAI embedding。需 OPENAI_API_KEY。默认 text-embedding-3-small(1536维)。

    按字符预算切小批 + 429 自动退避重试，适配低 TPM 账号（免费额度也不至于崩，只是慢）。
    """
    DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, model: str = "text-embedding-3-small",
                 char_budget: int = 16000, max_items: int = 128):
        try:
            import openai  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("需要 `pip install openai`") from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("未设置 OPENAI_API_KEY")
        import openai
        self._openai = openai
        self._client = openai.OpenAI()
        self.model = model
        self.char_budget = char_budget      # 近似 token 上限；默认 16k，稳在 40k TPM 内
        self.max_items = max_items
        # 可选降维（Matryoshka）：设 KB_RAG_OPENAI_DIM=512 让向量库/内存缩小到 1/3，检索质量几乎不掉。
        # ⚠️ 语料与查询必须用**同一** dim —— 建索引和起服务两边都要设成同一个值。
        req = int(os.environ.get("KB_RAG_OPENAI_DIM") or 0)
        full = self.DIMS.get(model, 1536)
        self._req_dims = req if (req and req < full) else None
        self.dim = self._req_dims or full

    def _one(self, batch: list[str], tries: int = 8) -> list[list[float]]:
        import time
        kw = {"dimensions": self._req_dims} if self._req_dims else {}
        for attempt in range(tries):
            try:
                resp = self._client.embeddings.create(model=self.model, input=batch, **kw)
                return [d.embedding for d in resp.data]
            except self._openai.RateLimitError:
                wait = min(5 * (2 ** attempt), 60)
                print(f"   限流(429)，等 {wait}s 重试…", flush=True)
                time.sleep(wait)
        raise RuntimeError("多次重试仍被限流——建议给 OpenAI 账号充值提高 TPM")

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for batch in _subbatches(texts, self.char_budget, self.max_items):
            out.extend(self._one(batch))
        return out


class VoyageEmbedder:
    """Voyage embedding（多语言/中文更强）。需 VOYAGE_API_KEY。默认 voyage-3。"""
    DIMS = {"voyage-3": 1024, "voyage-3-lite": 512}

    def __init__(self, model: str = "voyage-3", batch: int = 128,
                 input_type: str = "document"):
        try:
            import voyageai  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("需要 `pip install voyageai`") from e
        if not os.environ.get("VOYAGE_API_KEY"):
            raise RuntimeError("未设置 VOYAGE_API_KEY")
        import voyageai
        self._client = voyageai.Client()
        self.model = model
        self.batch = batch
        self.input_type = input_type
        self.dim = self.DIMS.get(model, 1024)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            chunk = [t or " " for t in texts[i:i + self.batch]]
            r = self._client.embed(chunk, model=self.model, input_type=self.input_type)
            out.extend(r.embeddings)
        return out


def _best_device() -> str:
    """自动选设备：Apple Silicon 用 mps、有 N 卡用 cuda、否则 cpu。可用 KB_RAG_LOCAL_DEVICE 覆盖。"""
    forced = os.environ.get("KB_RAG_LOCAL_DEVICE")
    if forced:
        return forced
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class LocalEmbedder:
    """本地句向量（sentence-transformers），**零 API、离线、免费**。默认 BAAI/bge-m3。

    bge-m3：多语言(100+ 语言)、跨语言检索强，1024 维——正合中文问英文书这类语料。
    首次会下载模型(约 2GB)到本地缓存，之后完全离线。设备自动选 mps/cuda/cpu（可用
    KB_RAG_LOCAL_DEVICE 覆盖）；CPU 上大模型较慢，Apple Silicon 会走 mps 提速。
    嫌 bge-m3 慢可换更小的：`KB_RAG_LOCAL_MODEL=intfloat/multilingual-e5-small`（384 维，快很多）。
    换模型维度变 → 需全量重嵌。语料与查询必须用同一模型。
    """
    def __init__(self, model: str | None = None, batch: int = 64):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("需要 `pip install sentence-transformers`（extra: local）") from e
        self.model_name = model or os.environ.get("KB_RAG_LOCAL_MODEL", "BAAI/bge-m3")
        self.device = _best_device()
        self._m = SentenceTransformer(self.model_name, device=self.device)
        self.batch = batch
        # 新版 sentence-transformers 把方法改名为 get_embedding_dimension；两者都兼容
        getdim = (getattr(self._m, "get_embedding_dimension", None)
                  or self._m.get_sentence_embedding_dimension)
        self.dim = int(getdim())

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._m.encode([t or " " for t in texts], batch_size=self.batch,
                              show_progress_bar=len(texts) > 8, convert_to_numpy=True,
                              normalize_embeddings=False)
        return [v.tolist() for v in vecs]


def make_embedder(name: str | None = None):
    """按名字/环境造 embedder。name: local|openai|voyage|fake；默认看有哪个 key。"""
    name = (name or os.environ.get("KB_RAG_EMBEDDER") or
            ("openai" if os.environ.get("OPENAI_API_KEY") else
             "voyage" if os.environ.get("VOYAGE_API_KEY") else "fake"))
    if name == "local":
        return LocalEmbedder()
    if name == "openai":
        return OpenAIEmbedder()
    if name == "voyage":
        return VoyageEmbedder()
    if name == "fake":
        return FakeEmbedder()
    raise ValueError(f"未知 embedder: {name}")
