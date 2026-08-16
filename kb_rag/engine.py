"""
engine.py — 确定性检索核 + 工具层（纯标准库，代码写死、可审计）。

这是 agent 的"手"：一组确定性工具。agent（LLM）只负责决定"调哪个、调几次、够没够"，
真正的分块/检索/取原文/引用核对全在这里，稳定可复现。

工具契约（agent 可调）：
    search(query, course=None, kind=None, k=8)  → 相关块 top-k（BM25+TFIDF·RRF）
    fetch_section(course, section)              → 整节 raw，按原文顺序（教学模式：不漏细节）
    outline(course)                             → 该课有序小节清单（核查覆盖用）
    list_courses()                              → 库里有哪些课
    verify(claim, chunk_ids)                    → 确定性引用核对：claim 是否被这些块支撑

语料来自 ingest.py 产出的 chunks（每块带 course/kind/section/order/layer 元数据）。
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

K1, B, RRF_K = 1.5, 0.75, 60
ASCII_TOK = re.compile(r"[a-z0-9]+")
CJK_RUN = re.compile(r"[一-鿿]+")


def tokenize(text: str) -> list[str]:
    """CJK bigram（单字兜底）+ 拉丁/数字词。无需 jieba。"""
    text = text.lower()
    toks = [t for t in ASCII_TOK.findall(text) if len(t) >= 2]
    for run in CJK_RUN.findall(text):
        toks.extend([run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)])
    return toks


@dataclass
class Chunk:
    id: str
    course: str
    kind: str          # course / video / book / data
    layer: str         # raw / synth
    section: str
    order: int         # 原文顺序（同一 section 内 + 全局递增）
    text: str


@dataclass
class Engine:
    chunks: list[Chunk]
    tf: list[dict] = field(default_factory=list)
    df: dict = field(default_factory=dict)
    idf_bm25: dict = field(default_factory=dict)
    idf_tfidf: dict = field(default_factory=dict)
    doc_len: list[int] = field(default_factory=list)
    avgdl: float = 0.0
    tfidf_norm: list[float] = field(default_factory=list)

    # ---------- 构建 ----------
    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Engine":
        chunks = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    chunks.append(Chunk(d["id"], d["course"], d["kind"], d["layer"],
                                        d["section"], d["order"], d["text"]))
        eng = cls(chunks=chunks)
        eng._index()
        return eng

    def _index(self) -> None:
        n = len(self.chunks)
        df: dict = defaultdict(int)
        for c in self.chunks:
            counts = Counter(tokenize(c.text + " " + c.section))
            self.tf.append(dict(counts))
            self.doc_len.append(sum(counts.values()))
            for t in counts:
                df[t] += 1
        self.df = dict(df)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        self.idf_bm25 = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.idf_tfidf = {t: math.log(n / d) + 1.0 for t, d in df.items()}
        for counts in self.tf:
            s = sum((f * self.idf_tfidf[t]) ** 2 for t, f in counts.items())
            self.tfidf_norm.append(math.sqrt(s) or 1.0)

    # ---------- 内部打分 ----------
    def _cands(self, q_terms, allow):
        allow = set(allow) if allow is not None else None
        out = set()
        for i, counts in enumerate(self.tf):
            if allow is not None and i not in allow:
                continue
            if any(t in counts for t in q_terms):
                out.add(i)
        return out

    def _bm25(self, q_terms, cand):
        sc = {}
        for i in cand:
            counts, dl, s = self.tf[i], self.doc_len[i], 0.0
            for t in q_terms:
                f = counts.get(t)
                if f:
                    s += self.idf_bm25.get(t, 0) * (f * (K1 + 1)) / (f + K1 * (1 - B + B * dl / self.avgdl))
            if s > 0:
                sc[i] = s
        return sorted(sc.items(), key=lambda x: -x[1])

    def _tfidf(self, q_terms, cand):
        qw = {t: c * self.idf_tfidf.get(t, 0) for t, c in Counter(q_terms).items()}
        qn = math.sqrt(sum(w * w for w in qw.values())) or 1.0
        sc = {}
        for i in cand:
            counts, dot = self.tf[i], 0.0
            for t, w in qw.items():
                f = counts.get(t)
                if f:
                    dot += w * (f * self.idf_tfidf[t])
            if dot > 0:
                sc[i] = dot / (self.tfidf_norm[i] * qn)
        return sorted(sc.items(), key=lambda x: -x[1])

    @staticmethod
    def _rrf(rankings, k=RRF_K):
        fused = defaultdict(float)
        for r in rankings:
            for rank, doc in enumerate(r):
                fused[doc] += 1.0 / (k + rank)
        return sorted(fused.items(), key=lambda x: -x[1])

    # ---------- 工具（agent 可调） ----------
    def search(self, query: str, course: str | None = None,
               kind: str | None = None, k: int = 8) -> list[dict]:
        q = tokenize(query)
        if not q:
            return []
        allow = None
        if course or kind:
            allow = [i for i, c in enumerate(self.chunks)
                     if (not course or course.lower() in c.course.lower())
                     and (not kind or kind == c.kind)]
            if not allow:
                return []
        cand = self._cands(q, allow)
        if not cand:
            return []
        bm = self._bm25(q, cand)
        tf = self._tfidf(q, cand)
        fused = self._rrf([[d for d, _ in bm], [d for d, _ in tf]])
        bm_top = {d: s for d, s in bm}
        out = []
        for i, rrf in fused[:k]:
            c = self.chunks[i]
            qs = set(q)
            cov = sum(1 for t in qs if t in self.tf[i]) / len(qs)
            out.append({"id": c.id, "course": c.course, "kind": c.kind,
                        "section": c.section, "layer": c.layer, "order": c.order,
                        "text": c.text, "rrf": round(rrf, 4),
                        "bm25": round(bm_top.get(i, 0), 2), "coverage": round(cov, 2)})
        return out

    def fetch_section(self, course: str, section: str) -> list[dict]:
        """整节按原文顺序取回——教学模式的"不漏细节"靠这个。"""
        got = [c for c in self.chunks
               if course.lower() in c.course.lower() and c.section == section]
        got.sort(key=lambda c: c.order)
        return [{"id": c.id, "course": c.course, "section": c.section,
                 "order": c.order, "text": c.text, "layer": c.layer} for c in got]

    def outline(self, course: str) -> list[str]:
        seen, out = set(), []
        for c in sorted((c for c in self.chunks if course.lower() in c.course.lower()),
                        key=lambda c: c.order):
            if c.section not in seen:
                seen.add(c.section)
                out.append(c.section)
        return out

    def list_courses(self) -> dict:
        cc: dict = defaultdict(int)
        for c in self.chunks:
            cc[c.course] += 1
        return dict(sorted(cc.items(), key=lambda x: -x[1]))

    def verify(self, claim: str, chunk_ids: list[str]) -> dict:
        """确定性引用核对：claim 的实义词有多少被指定块覆盖。防编造引用。"""
        by_id = {c.id: c for c in self.chunks}
        pool = " ".join(by_id[i].text for i in chunk_ids if i in by_id)
        pool_tokens = set(tokenize(pool))
        claim_tokens = set(tokenize(claim))
        if not claim_tokens:
            return {"supported": False, "coverage": 0.0, "reason": "claim 为空"}
        cov = len(claim_tokens & pool_tokens) / len(claim_tokens)
        return {"supported": cov >= 0.6, "coverage": round(cov, 2),
                "missing": sorted(claim_tokens - pool_tokens)[:12]}
