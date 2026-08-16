"""
ingest.py — 原始资料 → chunks.jsonl（带 course/kind/section/order/layer 元数据）。

定位（吸取 （一个更早的项目） 教训②：语料要对）：只吃 knowledge_base/ 下清洗过的**原始**资料
（transcript / 书 PDF 文本），打 layer=raw。你的 NotebookLM 拆解产物走另一条路（产出/自看），
不进这个池——保持 raw 池的忠实纯度。

目录约定（两种都支持）：
    A) 分类桶结构（推荐，配合 RAG-Database 仓库）：
        course/{课名}/*.{md,txt,vtt,srt}   ← kind=course
        book/{书名}/*.txt                  ← kind=book
        video/{课名}/*.txt                 ← kind=video
       —— 桶名决定 kind，桶内子文件夹名就是课名。
    B) 前缀平铺结构：
        {course}/*.{md,txt,vtt,srt}     ← kind=course（默认）
        Video_{course}/*.txt            ← kind=video
        Book_{name}/*.txt (或 pdf 抽出的 .txt) ← kind=book

分节（section）：
    - 书籍(kind=book)：优先按 [p.N] 页码窗口分节，节名 "p.a-b"（每 book_pages 页一节，
      尽量附章节提示如 "p.312-319 · 第18章 …"）。引用精确到页区间，便于回原书核对(防幻觉)。
      无页码标记(如失败的空转换) → 退回整篇一节。
    - 有 markdown ## 标题 → 按标题分节（清洗过的 transcript 建议先加小标题）
    - 纯文本无标题 → 整篇一节，节名=文件名
超长小节再按字符窗口切，带 overlap。

块 id = md5(相对路径 | 文件内局部序号)——**只依赖本文件内容**：加别的文件不改本文件 id，
改某本书只让那本书的 id 变。这样增量嵌入真正"只嵌新增/改动"，不会因插入而整库重嵌。
（order 字段仍是全局阅读序，供 fetch_section 排序，但不进 id。）

字幕清洗：.vtt/.srt 自动去时间轴/序号/WEBVTT 头，合并为纯文本。
PDF：本模块只吃文本；PDF 请先用 pdftotext/pymupdf 抽成 .txt 再放进来（保持 ingest 纯 stdlib）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

TS = re.compile(r"^\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->")     # srt/vtt 时间轴
CUE_IDX = re.compile(r"^\d+\s*$")                          # srt 序号行
HEADING = re.compile(r"^#{1,6}\s+(.*)")

# 书籍分节用：pdf2txt/OCR 产出的页码标记 [p.N]（独占一行）。47/57 本书都有，作通用切分锚点。
PAGE = re.compile(r"^[ \t]*\[p\.(\d+)\][ \t]*$", re.MULTILINE)
# 章节提示（仅用于给 p.a-b 节名附一句可读上下文，非切分依据，宁松勿严）：
CHAP = re.compile(
    r"第\s*[0-9]{1,3}\s*[章回]"
    r"|第\s*[一二三四五六七八九十百零]{1,7}\s*[章回]"
    r"|第\s*[0-9一二三四五六七八九十]{1,7}\s*[部篇卷]"
    r"|\bChapter\s+\d{1,3}\b"
    r"|\bPart\s+(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|[IVXLC]{1,6}|\d{1,2})\b",
    re.IGNORECASE,
)


def _chapter_hint(text: str) -> str | None:
    """在一段文本里找第一条像"章节标题"的短行，作节名的可读补充。找不到返回 None。"""
    for ln in text.splitlines():
        s = ln.strip()
        # 只认"章节标记就在行首"的短行（真标题），排除正文里的交叉引用/句中提及
        if s and len(s) <= 60 and CHAP.match(s):
            return s[:48]
    return None


def book_sections(text: str, pages_per: int) -> list[tuple[str, str]]:
    """书籍按页码窗口分节：每 pages_per 个 [p.N] 页为一节，节名 'p.a-b'（尽量附章节提示）。

    为什么按页而非按章：语料是多语言 + OCR，章节标记五花八门且常被 OCR 打花，按页码最鲁棒、
    100% 覆盖，且引用能精确到页码区间（便于回原书核对，正合"防幻觉"诉求）。
    无页码标记（如失败的空转换）→ 返回 []，交回 sections_of 走整篇兜底。
    """
    pages = [(int(m.group(1)), m.start()) for m in PAGE.finditer(text)]
    if len(pages) < 2:
        return []
    out: list[tuple[str, str]] = []
    for i in range(0, len(pages), pages_per):
        grp = pages[i:i + pages_per]
        start = grp[0][1]
        end = pages[i + pages_per][1] if i + pages_per < len(pages) else len(text)
        body = text[start:end].strip()
        if len(body) < 40:
            continue
        a, b = grp[0][0], grp[-1][0]
        title = f"p.{a}-{b}" if a != b else f"p.{a}"
        hint = _chapter_hint(body)
        if hint:
            title = f"{title} · {hint}"
        out.append((title, body))
    return out


def clean_subtitle(text: str) -> str:
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s == "WEBVTT" or TS.match(s) or CUE_IDX.match(s):
            continue
        s = re.sub(r"<[^>]+>", "", s)          # 去内联标签
        out.append(s)
    # 去相邻重复（自动字幕常整行重复）
    dedup = []
    for s in out:
        if not dedup or dedup[-1] != s:
            dedup.append(s)
    return "\n".join(dedup)


BUCKETS = {"course": "course", "book": "book", "video": "video", "data": "data"}


def kind_of(folder: str) -> tuple[str, str]:
    if folder.startswith("Video_"):
        return "video", folder[6:].replace("_", " ")
    if folder.startswith("Book_"):
        return "book", folder[5:].replace("_", " ")
    if folder.startswith("Data_"):
        return "data", folder[5:].replace("_", " ")
    return "course", folder.replace("_", " ")


def classify(parts: tuple[str, ...]) -> tuple[str, str, str | None] | None:
    """由相对路径判定 (kind, course, series)。支持分类桶与前缀平铺两种。

    分类桶：course/book/video/data 作顶层桶。
        桶/课名/文件            → course=课名, series=None
        桶/系列/…/课名/文件     → course=最深层文件夹, series=中间层(可多级用 / 连)
    课名恒取“直接放该文件的那层文件夹”，中间任意层都算系列/分组。
    """
    # A) 分类桶
    if parts[0].lower() in BUCKETS:
        if len(parts) < 3:
            return None  # 桶下直接放文件、缺课名子文件夹 → 跳过
        kind = BUCKETS[parts[0].lower()]
        course = parts[-2].replace("_", " ")           # 直接父文件夹 = 课名
        mids = parts[1:-2]                              # 桶与课名之间的层 = 系列
        series = "/".join(m.replace("_", " ") for m in mids) if mids else None
        return kind, course, series
    # B) 前缀平铺：需二级深度
    if len(parts) < 2:
        return None
    kind, course = kind_of(parts[0])
    return kind, course, None


def window(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    step = max(1, max_chars - overlap)
    return [text[i:i + max_chars] for i in range(0, len(text), step)]


def sections_of(text: str, default_title: str) -> list[tuple[str, str]]:
    """有 ## 标题按标题分节；否则**整篇作为一节**，节名用 default_title（文件名）。

    无标题的字幕/transcript：一个文件 = 一节课，节名唯一（文件名），
    这样 teach 的 fetch_section 能精确取回某一节课的全部内容、按序不混。
    超长节由上层 window() 再切成多块，同属该节、order 递增。
    """
    if HEADING.search(text):
        parts, title, cur = [], "（引言）", []
        for ln in text.splitlines():
            m = HEADING.match(ln)
            if m:
                if "".join(cur).strip():
                    parts.append((title, "\n".join(cur).strip()))
                title, cur = m.group(1).strip(), []
            else:
                cur.append(ln)
        if "".join(cur).strip():
            parts.append((title, "\n".join(cur).strip()))
        return parts
    # 无标题：整篇一节，节名 = 文件名（不再 seg-0001 撞名）
    body = text.strip()
    return [(default_title, body)] if body else []


def run(src: Path, out: Path, layer: str, max_chars: int, overlap: int,
        book_pages: int = 8) -> None:
    order = 0
    n_files = n_chunks = 0
    per: dict = {}
    with open(out, "w", encoding="utf-8") as fo:
        for path in sorted(src.rglob("*")):
            if path.suffix.lower() not in (".md", ".txt", ".vtt", ".srt"):
                continue
            rel = path.relative_to(src)
            if path.name.startswith("_"):
                continue
            kc = classify(rel.parts)
            if kc is None:
                continue
            kind, course, series = kc
            raw = path.read_text(encoding="utf-8", errors="ignore")
            if path.suffix.lower() in (".vtt", ".srt"):
                raw = clean_subtitle(raw)
            n_files += 1
            # 书籍：优先按页码窗口分节（引用精确到页）；无页码标记再退回整篇。
            secs = book_sections(raw, book_pages) if kind == "book" else None
            if not secs:
                secs = sections_of(raw, path.stem)
            fidx = 0   # 文件内局部序号：进 id，使 id 只依赖本文件内容，加别的文件不改本文件 id
            for sec_title, sec_text in secs:
                for piece in window(sec_text, max_chars, overlap):
                    piece = piece.strip()
                    if len(piece) < 40:
                        continue
                    order += 1     # 全局阅读顺序（进 order 字段，供 fetch_section 排序，不进 id）
                    fidx += 1
                    cid = hashlib.md5(f"{rel}|{fidx}".encode()).hexdigest()[:12]
                    fo.write(json.dumps({
                        "id": cid, "course": course, "kind": kind, "layer": layer,
                        "series": series, "section": sec_title, "order": order,
                        "text": piece,
                    }, ensure_ascii=False) + "\n")
                    n_chunks += 1
                    per[course] = per.get(course, 0) + 1
    print(f"✅ ingest：{n_files} 文件 → {n_chunks} 块（layer={layer}）→ {out}")
    for c, k in sorted(per.items(), key=lambda x: -x[1]):
        print(f"   {k:5d}  {c}")


def main() -> None:
    ap = argparse.ArgumentParser(description="原始资料 → chunks.jsonl")
    ap.add_argument("--src", type=Path, required=True, help="knowledge_base/raw 根目录")
    ap.add_argument("--out", type=Path, default=Path("chunks.jsonl"))
    ap.add_argument("--layer", default="raw", choices=["raw", "synth"])
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--book-pages", type=int, default=8,
                    help="书籍每节多少页（按 [p.N] 页码窗口分节，引用精确到页区间）")
    a = ap.parse_args()
    if not a.src.is_dir():
        print(f"✗ 找不到目录：{a.src}", file=sys.stderr)
        sys.exit(1)
    run(a.src, a.out, a.layer, a.max_chars, a.overlap, a.book_pages)


if __name__ == "__main__":
    main()
