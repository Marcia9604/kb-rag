"""
shrinkvecs.py — 把 OpenAI 向量库降维 + 半精度存储，缩到能过 `railway up` 上传上限。

原理：OpenAI text-embedding-3 系列是 **Matryoshka（套娃）** 向量——前 N 维已承载绝大部分语义。
所以把 1536 维**截断到前 384 维 → 重新归一化 → 存成 float16**，文件缩到 ~1/8，检索质量几乎不掉。
查询侧只要设 `KB_RAG_OPENAI_DIM=384`（OpenAI 直接返回 384 维），两边维度对齐即可。

用法（本机，一次性，免费、几秒）：
    python -m kb_rag.shrinkvecs --in vectors-openai.npz --out vectors-openai-384.npz --dim 384

- 输入是你已建好的 1536 维 `vectors-openai.npz`（不改动它，留着备用）。
- 输出 `vectors-openai-384.npz` 就是要上云的那份（~95MB）。
- 之后云端 `KB_RAG_OPENAI_DIM` 必须设成同一个 `--dim`（默认 384）。
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    import numpy as np
    ap = argparse.ArgumentParser(description="降维+半精度压缩 OpenAI 向量库以便上云")
    ap.add_argument("--in", dest="inp", type=Path, default=Path("vectors-openai.npz"))
    ap.add_argument("--out", type=Path, default=Path("vectors-openai-384.npz"))
    ap.add_argument("--dim", type=int, default=384, help="截断到前 N 维（须与云端 KB_RAG_OPENAI_DIM 一致）")
    ap.add_argument("--fp32", action="store_true", help="保留 float32（默认存 float16，更小）")
    a = ap.parse_args()

    if not a.inp.exists():
        raise SystemExit(f"❌ 找不到 {a.inp}；先跑 buildindex 生成 OpenAI 向量库")
    d = np.load(a.inp, allow_pickle=False)
    ids, mat = d["ids"], d["mat"]
    src_shape, src_dim = mat.shape, mat.shape[1]
    if a.dim > src_dim:
        raise SystemExit(f"❌ --dim {a.dim} 大于原维度 {src_dim}，只能截断变小")

    mat = mat[:, :a.dim].astype("float32")                 # 截断到前 dim 维
    n = np.linalg.norm(mat, axis=1, keepdims=True)         # 重新归一化（截断后不再是单位向量）
    n[n == 0] = 1.0
    mat = mat / n
    if not a.fp32:
        mat = mat.astype("float16")                        # 半精度：文件再砍一半，余弦几乎无损

    np.savez(a.out, ids=ids, mat=mat)
    size_mb = a.out.stat().st_size / 1e6
    print(f"✅ {a.inp} {src_shape} float32 → {a.out} {mat.shape} {mat.dtype} "
          f"（{size_mb:.0f} MB）")
    print(f"   记得云端设 KB_RAG_OPENAI_DIM={a.dim}（和这里一致），否则维度对不上会崩。")


if __name__ == "__main__":
    main()
