"""最小冒烟测试（纯 stdlib，跑 mock 后端）。断言 agentic 回路的关键行为。"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kb_rag.ingest import run as ingest_run          # noqa: E402
from kb_rag.engine import Engine                     # noqa: E402
from kb_rag.agent import Agent, MockBackend          # noqa: E402


def build_agent(tmp: Path) -> Agent:
    out = tmp / "chunks.jsonl"
    ingest_run(ROOT / "tests" / "fixture", out, "raw", 1200, 120)
    return Agent(Engine.from_jsonl(out), MockBackend())


class Smoke(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.ag = build_agent(Path(self._d.name))

    def tearDown(self):
        self._d.cleanup()

    def test_ingest_has_layer_and_order(self):
        rows = [json.loads(l) for l in (Path(self._d.name) / "chunks.jsonl").read_text().splitlines()]
        self.assertTrue(rows)
        self.assertTrue(all(r["layer"] == "raw" for r in rows))
        self.assertEqual([r["order"] for r in rows], sorted(r["order"] for r in rows))

    def test_search_hits_with_citation(self):
        a = self.ag.ask("temperature 随机性")
        self.assertFalse(a.blocked)
        self.assertTrue(a.citations)

    def test_out_of_library_is_blocked(self):
        a = self.ag.ask("番茄炒蛋怎么做放几个鸡蛋")
        self.assertTrue(a.blocked)

    def test_search_loop_iterates(self):
        # 覆盖不足时应多轮迭代（agentic 证据），不是一次性
        a = self.ag.ask("temperature 什么时候用低什么时候用高更合适一些")
        self.assertGreaterEqual(a.rounds, 1)
        self.assertTrue(any("round" in t for t in a.trace))

    def test_teach_fetches_ordered_section(self):
        a = self.ag._teach("监督学习 映射 标签", "Sample")
        self.assertEqual(a.mode, "teach")
        self.assertTrue(a.citations)
        # 引用应集中在同一小节（整节取回）
        self.assertEqual(len({c["section"] for c in a.citations}), 1)


class BookSectioning(unittest.TestCase):
    """书籍按 [p.N] 页码分节 + 块 id 只依赖本文件（增量嵌入不整库重嵌）。"""

    def _book(self, n_pages: int) -> str:
        body = "这是正文内容，用于测试页码分节。" * 8
        return "\n".join(f"[p.{i}]\n{body}" for i in range(1, n_pages + 1))

    def test_page_window_sections(self):
        from kb_rag.ingest import book_sections
        secs = book_sections(self._book(20), pages_per=8)
        titles = [t for t, _ in secs]
        self.assertEqual(titles[0], "p.1-8")
        self.assertEqual(titles[1], "p.9-16")
        self.assertTrue(len(secs) >= 3)         # 20 页 / 8 → 3 节

    def test_no_page_marks_falls_back(self):
        from kb_rag.ingest import book_sections
        self.assertEqual(book_sections("没有页码标记的纯文本。" * 20, 8), [])

    def test_ids_independent_of_other_files(self):
        # 加一本新书，老文件 id 一个都不能变（否则增量嵌入会整库重嵌）
        def ids(src: Path, out: Path):
            ingest_run(src, out, "raw", 1200, 120)
            return {json.loads(l)["id"] for l in out.read_text().splitlines()}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "db" / "book" / "书A").mkdir(parents=True)
            (d / "db" / "book" / "书A" / "书A.txt").write_text(self._book(12), encoding="utf-8")
            before = ids(d / "db", d / "a.jsonl")
            (d / "db" / "book" / "书B").mkdir(parents=True)
            (d / "db" / "book" / "书B" / "书B.txt").write_text(self._book(12), encoding="utf-8")
            after = ids(d / "db", d / "b.jsonl")
            self.assertEqual(before - after, set())     # 老 id 无丢失
            self.assertTrue(after - before)             # 新书产出新 id


class VerifyGuard(unittest.TestCase):
    """本地 verify 护栏：真实 id + 有支撑 → SUPPORTED；编造 id → BLOCK。"""

    def _chunks(self, tmp: Path) -> Path:
        out = tmp / "chunks.jsonl"
        ingest_run(ROOT / "tests" / "fixture", out, "raw", 1200, 120)
        return out

    def test_supported_and_block(self):
        from kb_rag.engine import Engine
        from kb_rag.retrieve import cmd_verify
        import argparse
        with tempfile.TemporaryDirectory() as d:
            eng = Engine.from_jsonl(self._chunks(Path(d)))
            real_id = eng.chunks[0].id
            claim = eng.chunks[0].text[:60]
            # 有支撑 → 不退出（SUPPORTED）
            ns = argparse.Namespace(claim=claim, chunk_ids=[real_id], func=cmd_verify)
            cmd_verify(eng, ns)          # 不应抛 SystemExit
            # 编造 id → BLOCK（SystemExit 2）
            ns2 = argparse.Namespace(claim="x", chunk_ids=["deadbeef0000"], func=cmd_verify)
            with self.assertRaises(SystemExit) as cm:
                cmd_verify(eng, ns2)
            self.assertEqual(cm.exception.code, 2)


class GoldenEval(unittest.TestCase):
    """黄金集评估的匹配/排名/指标逻辑。"""

    def test_hit_matching(self):
        from kb_rag.eval import _hit_matches
        hit = {"kind": "book", "course": "思考，快与慢", "section": "p.131-138 · 第11章 锚定效应"}
        self.assertTrue(_hit_matches(hit, {"kind": "book", "section_contains": "锚定"}))
        self.assertTrue(_hit_matches(hit, {"course": "思考"}))
        self.assertFalse(_hit_matches(hit, {"kind": "course"}))
        self.assertFalse(_hit_matches(hit, {"section_contains": "禀赋效应"}))
        self.assertFalse(_hit_matches(hit, {}))          # 没给定位字段不算命中

    def test_rank_and_metrics(self):
        from kb_rag.eval import _rank_of_first_match, _metrics
        hits = [{"course": "A", "section": "x"}, {"course": "B", "section": "锚定"}]
        self.assertEqual(_rank_of_first_match(hits, [{"section_contains": "锚定"}]), 2)
        self.assertEqual(_rank_of_first_match(hits, [{"course": "Z"}]), 0)
        m = _metrics([1, 2, 0], k=8)
        self.assertEqual(m["n"], 3)
        self.assertAlmostEqual(m["recall@1"], 1 / 3)
        self.assertAlmostEqual(m["recall@3"], 2 / 3)
        self.assertAlmostEqual(m["mrr"], (1.0 + 0.5 + 0.0) / 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
