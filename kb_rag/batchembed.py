"""
batchembed.py — 大规模嵌入用 OpenAI Batch API：**提交后关掉终端，稍后取回**，不用在终端一点点熬。

优势（数据大时）：异步、便宜 50%、不撞每分钟速率限、一次提交几万块。**增量**：只提交没嵌过的块。

两步（不用守着）：
    python -m kb_rag.batchembed submit --chunks chunks.jsonl --out vectors.npz
      → 提交后打印 batch id，可以关终端/去睡觉（通常几分钟~1小时完成，最长 24h）
    python -m kb_rag.batchembed fetch  --out vectors.npz
      → 回来取回，写进向量库；没好就提示再等

对比：`kb_rag.buildindex` 是同步的（适合少量新增，几千块内）；数据大就用本 batch 流程。
状态记在 <out>.batch.json。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .embed import OpenAIEmbedder
from .engine import Engine
from .vecstore import EmbeddingStore

MAX_LINES = 40000   # 每个 batch 输入文件的请求行数上限（OpenAI 上限 5 万，留余量）


def _req_line(cid: str, text: str, model: str) -> str:
    return json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/embeddings",
                       "body": {"model": model, "input": text or " "}}, ensure_ascii=False)


def _parse_output(text: str) -> dict:
    """batch 输出文件文本 → {custom_id: embedding}。跳过出错行。"""
    out = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("error") or o.get("response", {}).get("status_code") != 200:
            continue
        out[o["custom_id"]] = o["response"]["body"]["data"][0]["embedding"]
    return out


def _client():
    try:
        import openai
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("需要 `pip install openai`") from e
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("未设置 OPENAI_API_KEY")
    import openai
    return openai.OpenAI()


def _statef(out: Path) -> Path:
    return Path(str(out) + ".batch.json")


def submit(chunks: Path, out: Path, model: str) -> None:
    client = _client()
    eng = Engine.from_jsonl(chunks)
    dim = OpenAIEmbedder.DIMS.get(model, 1536)
    store = EmbeddingStore.load(out) if out.exists() else EmbeddingStore.empty(dim)
    missing = [c for c in eng.chunks if not store.has(c.id)]
    print(f"块数 {len(eng.chunks)} | 已有 {len(eng.chunks) - len(missing)} | 需嵌 {len(missing)}")
    if not missing:
        print("✅ 全部已嵌，无需提交")
        return

    batches = []
    for i in range(0, len(missing), MAX_LINES):
        shard = missing[i:i + MAX_LINES]
        req = Path(f"{out}.req.{i // MAX_LINES}.jsonl")
        req.write_text("\n".join(_req_line(c.id, c.text, model) for c in shard),
                       encoding="utf-8")
        up = client.files.create(file=open(req, "rb"), purpose="batch")
        b = client.batches.create(input_file_id=up.id, endpoint="/v1/embeddings",
                                  completion_window="24h")
        batches.append(b.id)
        req.unlink(missing_ok=True)
        print(f"  提交 shard {i // MAX_LINES}: {len(shard)} 块 → batch {b.id}")

    _statef(out).write_text(json.dumps({"out": str(out), "model": model, "batches": batches}))
    print(f"✅ 已提交 {len(missing)} 块，共 {len(batches)} 个 batch。可关终端，稍后："
          f"\n   python -m kb_rag.batchembed fetch --out {out}")


def fetch(out: Path) -> None:
    client = _client()
    sf = _statef(out)
    if not sf.exists():
        print(f"✗ 找不到状态文件 {sf}，先 submit")
        return
    st = json.loads(sf.read_text())
    model, batch_ids = st["model"], st["batches"]

    results, done, failed = {}, True, False
    for bid in batch_ids:
        b = client.batches.retrieve(bid)
        print(f"  batch {bid}: {b.status} | {getattr(b, 'request_counts', '')}")
        if b.status in ("failed", "expired", "cancelled"):
            failed = True
            errs = getattr(getattr(b, "errors", None), "data", None) or []
            for e in errs:
                print(f"    ✗ {getattr(e, 'code', '?')}: {getattr(e, 'message', e)}")
            continue
        if b.status != "completed":
            done = False
            continue
        content = client.files.content(b.output_file_id).text
        results.update(_parse_output(content))
    if failed:
        print("✗ 有 batch 失败/过期。若是 token_limit_exceeded：Batch 排队 token 有上限"
              "（本组织 3M），大库改用同步 `python -m kb_rag.buildindex`（带退避、增量、逐批落盘）。"
              "\n  修因后重新 submit（本次失败不产生费用）。")
        return
    if not done:
        print("⏳ 还有 batch 没完成，过会儿再 fetch（通常几分钟~1小时）")
        return

    dim = OpenAIEmbedder.DIMS.get(model, 1536)
    store = EmbeddingStore.load(out) if out.exists() else EmbeddingStore.empty(dim)
    ids = list(results)
    store.add(ids, [results[i] for i in ids])
    store.save(out)
    sf.unlink(missing_ok=True)
    print(f"✅ 取回 {len(results)} 块 → {out}（向量库共 {store.mat.shape[0]} 块）")


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenAI Batch API 批量嵌入（异步、便宜、不守终端）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("submit", help="提交批量嵌入任务")
    ps.add_argument("--chunks", type=Path, default=Path("chunks.jsonl"))
    ps.add_argument("--out", type=Path, default=Path("vectors.npz"))
    ps.add_argument("--model", default="text-embedding-3-small")
    pf = sub.add_parser("fetch", help="取回已完成的批量结果")
    pf.add_argument("--out", type=Path, default=Path("vectors.npz"))
    a = ap.parse_args()
    if a.cmd == "submit":
        submit(a.chunks, a.out, a.model)
    else:
        fetch(a.out)


if __name__ == "__main__":
    main()
