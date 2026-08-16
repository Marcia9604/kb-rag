"""
vecstore.py — 向量存储（按 chunk id 存），支持增量嵌入。

块 id 是内容哈希（稳定）。扩充数据时只嵌新 id、复用旧向量，不再全量重嵌。
存为单个 .npz：ids(字符串数组) + mat(N×dim, 已归一化)。
"""
from __future__ import annotations

from pathlib import Path


class EmbeddingStore:
    def __init__(self, ids, mat):
        import numpy as np
        self.mat = np.asarray(mat, dtype="float32")
        self.ids = list(ids)
        self._i = {cid: i for i, cid in enumerate(self.ids)}

    @classmethod
    def empty(cls, dim: int) -> "EmbeddingStore":
        import numpy as np
        return cls([], np.zeros((0, dim), dtype="float32"))

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingStore":
        import numpy as np
        d = np.load(path, allow_pickle=False)
        return cls([str(x) for x in d["ids"]], d["mat"])

    def save(self, path: str | Path) -> None:
        import numpy as np
        np.savez(path, ids=np.array(self.ids, dtype=object).astype("U"), mat=self.mat)

    def has(self, cid: str) -> bool:
        return cid in self._i

    @staticmethod
    def _normalize(mat):
        import numpy as np
        mat = np.asarray(mat, dtype="float32")
        n = np.linalg.norm(mat, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return (mat / n).astype("float32")

    def add(self, ids: list[str], vecs) -> None:
        """追加新块（vecs 会被归一化）。已存在的 id 跳过。"""
        import numpy as np
        new_ids, new_rows = [], []
        vecs = self._normalize(vecs)
        for cid, row in zip(ids, vecs):
            if cid in self._i:
                continue
            self._i[cid] = len(self.ids) + len(new_ids)
            new_ids.append(cid)
            new_rows.append(row)
        if new_ids:
            self.ids.extend(new_ids)
            self.mat = (np.vstack([self.mat, np.asarray(new_rows, dtype="float32")])
                        if self.mat.size else np.asarray(new_rows, dtype="float32"))

    def matrix_for(self, ids: list[str]):
        """按给定 id 顺序拼出对齐矩阵。缺任何 id 报错（提示先补嵌）。"""
        import numpy as np
        missing = [cid for cid in ids if cid not in self._i]
        if missing:
            raise KeyError(f"{len(missing)} 个块尚无向量（需先 buildindex 补嵌），"
                           f"例如 {missing[:2]}")
        return np.asarray([self.mat[self._i[cid]] for cid in ids], dtype="float32")
