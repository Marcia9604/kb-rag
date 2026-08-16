"""
pdf2txt.py — 全自动 PDF → txt（book 语料）。自己识别文字版/扫描版/混合版，该 OCR 才 OCR。

核心 RAG 不依赖它，单独跑。为什么本地抽：仓库只存文本、不存大 PDF —— 绕开 GitHub
25MB 网页上传限制、不撑爆仓库。抽出的 txt 放进 book/<书名>/ 再提交即可。

自动判别（无需你指定类型）：
  1. 先用 PyMuPDF 逐页抽"文字层"。
  2. 统计有字页比例 + 总字数：
       文字充足           → 文字版，直接用（快、质量最好）
       文字过少/大量图片页 → 扫描或混合版，调 ocrmypdf --skip-text 只 OCR 图片页
         （--skip-text 保留已有文字页不动，只认图片页，混合版也正确）
  3. 混合版（部分页有字、部分是图）自动只补 OCR 缺字的页。

每页正文前插 `[p.N]` 页码标记（便于以后引用到页、按页分节）。--no-pages 关掉。

依赖：
    pip install pymupdf
    brew install ocrmypdf tesseract-lang     # 仅扫描/混合版需要 OCR 时

用法：
    python -m kb_rag.pdf2txt <书.pdf> [更多.pdf ...]      # 每个 → 同名 .txt
    python -m kb_rag.pdf2txt *.pdf                         # 批量
    python -m kb_rag.pdf2txt 书.pdf -o "book/如何阅读一本书/书.txt"
    python -m kb_rag.pdf2txt 书.pdf --lang chi_tra+eng     # 繁体
    python -m kb_rag.pdf2txt 书.pdf --ocr force            # 强制整本 OCR
    python -m kb_rag.pdf2txt 书.pdf --ocr off              # 禁止 OCR（只抽文字层）

小技巧（Mac）：把 PDF 从访达拖进终端，自动填好完整路径。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MIN_CHARS_PER_PAGE = 10      # 单页 ≥ 该字数算"有字页"
OCR_TEXT_RATIO = 0.6         # 有字页比例 < 此值 → 判为扫描/混合，需 OCR
OCR_MIN_TOTAL = 500          # 或全书总字数 < 此值 → 需 OCR（兜住"只有元数据页"的扫描书）


def _pages_text(doc, pages: bool) -> tuple[list[str], int, int]:
    """逐页抽文字层。返回 (每页文本列表, 有字页数, 总字数)。空页保留占位以维持页号。"""
    out, n_text, total = [], 0, 0
    for i, page in enumerate(doc, 1):
        txt = page.get_text().strip()
        if len(txt) >= MIN_CHARS_PER_PAGE:
            n_text += 1
            total += len(txt)
        out.append((f"[p.{i}]\n{txt}" if pages else txt) if txt else "")
    return out, n_text, total


def _join(chunks: list[str]) -> str:
    return "\n\n".join(c for c in chunks if c.strip())


def _run_ocr(pdf: Path, lang: str, force: bool) -> Path:
    """调 ocrmypdf 生成带文字层的新 PDF，返回其路径（临时文件）。"""
    if shutil.which("ocrmypdf") is None:
        raise RuntimeError("需要 OCR 但未装 ocrmypdf —— 运行：brew install ocrmypdf tesseract-lang")
    tmp = Path(tempfile.mkdtemp()) / (pdf.stem + "_ocr.pdf")
    mode = "--force-ocr" if force else "--skip-text"   # skip-text 只 OCR 图片页，保留文字页
    cmd = ["ocrmypdf", mode, "-l", lang, str(pdf), str(tmp)]
    print(f"   OCR（{mode} -l {lang}）…", flush=True)
    subprocess.run(cmd, check=True)
    return tmp


def extract(pdf: Path, lang: str = "chi_sim+eng", pages: bool = True,
            ocr: str = "auto") -> tuple[str, str]:
    """抽取一个 PDF。返回 (文本, 状态说明)。ocr: auto|force|off。"""
    try:
        import fitz  # PyMuPDF
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("需要 `pip install pymupdf`") from e

    doc = fitz.open(pdf)
    if doc.needs_pass:
        doc.close()
        return "", "加密 PDF，跳过（需先解密）"

    chunks, n_text, total = _pages_text(doc, pages)
    n_pages = doc.page_count
    ratio = (n_text / n_pages) if n_pages else 0.0
    doc.close()

    need_ocr = ocr == "force" or (
        ocr == "auto" and (ratio < OCR_TEXT_RATIO or total < OCR_MIN_TOTAL))
    if ocr == "off":
        need_ocr = False

    if not need_ocr:
        kind = "文字版" if ocr != "off" else "仅抽文字层（--ocr off）"
        return _join(chunks), f"{kind}：{n_text}/{n_pages} 页有字，{total:,} 字"

    # 需要 OCR：跑 ocrmypdf 后重新抽
    forced = (ocr == "force")
    ocr_pdf = _run_ocr(pdf, lang, force=forced)
    import fitz
    d2 = fitz.open(ocr_pdf)
    chunks2, n_text2, total2 = _pages_text(d2, pages)
    d2.close()
    shutil.rmtree(ocr_pdf.parent, ignore_errors=True)

    # 兜底：auto 走的是 --skip-text（只 OCR 无字页）。若扫描书每页有水印/乱码"假文字"，
    # 这些页会被跳过 → 结果近乎空。此时自动改用 --force-ocr 整本重 OCR。
    if not forced and total2 < OCR_MIN_TOTAL:
        print(f"   ⚠ skip-text 只抽到 {total2:,} 字，疑似假文字层，改用 --force-ocr 整本重 OCR…",
              flush=True)
        ocr_pdf = _run_ocr(pdf, lang, force=True)
        d3 = fitz.open(ocr_pdf)
        chunks2, n_text2, total2 = _pages_text(d3, pages)
        d3.close()
        shutil.rmtree(ocr_pdf.parent, ignore_errors=True)
        return _join(chunks2), f"扫描版(假文字层) → 强制OCR：{n_text2}/{n_pages} 页有字，{total2:,} 字"

    tag = "扫描版" if ratio < 0.1 else "混合版"
    return _join(chunks2), f"{tag} → OCR：{n_text2}/{n_pages} 页有字，{total2:,} 字"


def main() -> None:
    ap = argparse.ArgumentParser(description="全自动 PDF → txt（自动识别文字版/扫描版/混合版）")
    ap.add_argument("pdfs", nargs="+", type=Path, help="一个或多个 PDF")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="输出路径（仅单个输入时用）；默认与 PDF 同目录同名 .txt")
    ap.add_argument("--lang", default="chi_sim+eng",
                    help="OCR 语言（默认 chi_sim+eng；繁体 chi_tra，纯英文 eng）")
    ap.add_argument("--ocr", choices=["auto", "force", "off"], default="auto",
                    help="auto=自动判别(默认)；force=强制整本 OCR；off=只抽文字层不 OCR")
    ap.add_argument("--no-pages", action="store_true", help="不插入 [p.N] 页码标记")
    a = ap.parse_args()
    if a.out and len(a.pdfs) > 1:
        print("✗ -o 只能配单个 PDF", file=sys.stderr)
        sys.exit(1)
    rc = 0
    for pdf in a.pdfs:
        if not pdf.is_file():
            print(f"✗ 找不到：{pdf}", file=sys.stderr)
            rc = 1
            continue
        print(f"▶ {pdf.name}", flush=True)
        try:
            text, status = extract(pdf, lang=a.lang, pages=not a.no_pages, ocr=a.ocr)
        except Exception as e:
            print(f"   ✗ 失败：{type(e).__name__}: {e}", file=sys.stderr)
            rc = 1
            continue
        if not text.strip():
            print(f"   ⚠ 未抽到文本（{status}）", file=sys.stderr)
            rc = 1
            continue
        out = a.out or pdf.with_suffix(".txt")
        out.write_text(text, encoding="utf-8")
        print(f"   ✅ {status} → {out}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
