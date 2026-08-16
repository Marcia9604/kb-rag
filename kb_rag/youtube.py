"""
youtube.py — 把 YouTube 视频/播放列表的**字幕**抓成文件，放进 video/课名/，供 ingest 收录。

RAG 吃的是文字，所以这里只抓字幕（优先人工字幕，没有就用自动字幕），**不下载视频**。
底层用 yt-dlp（成熟、能处理播放列表/频道、边界情况多）。抓下来的 .vtt 交给 ingest 清洗
（去时间轴/去重），无需你手动处理。

依赖：
    pip install yt-dlp          # 或 pip install -e ".[youtube]"

用法：
    python -m kb_rag.youtube "视频URL" --course "课名"
    python -m kb_rag.youtube "播放列表URL" --course "课名"        # 整个列表一次抓
    python -m kb_rag.youtube "URL1" "URL2" --course "课名" --series "系列名"
    python -m kb_rag.youtube "URL" --course "课名" --lang "en,zh-Hans"   # 指定字幕语言
    python -m kb_rag.youtube "URL" --course "课名" --root ~/RAG-Database

抓完跑 ./update.sh 入库（只嵌新增）。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def fetch(urls: list[str], course: str, root: Path, series: str | None,
          lang: str) -> int:
    if shutil.which("yt-dlp") is None:
        print("✗ 未装 yt-dlp —— 运行：pip install yt-dlp", file=sys.stderr)
        return 1
    outdir = root / "video"
    if series:
        outdir = outdir / series
    outdir = outdir / course
    outdir.mkdir(parents=True, exist_ok=True)

    # 只抓字幕、不下视频；优先人工字幕(--write-subs)，退回自动字幕(--write-auto-subs)；
    # 统一转成 vtt；文件名用视频标题（ingest 会把文件名当节名）。
    cmd = [
        "yt-dlp", "--skip-download",
        "--write-subs", "--write-auto-subs",
        "--sub-langs", lang, "--sub-format", "vtt", "--convert-subs", "vtt",
        "--ignore-errors",
        "-o", str(outdir / "%(title)s.%(ext)s"),
        *urls,
    ]
    print(f"▶ 抓字幕 → {outdir}（语言 {lang}）")
    subprocess.run(cmd, check=False)

    got = sorted(outdir.glob("*.vtt"))
    print(f"\n✅ {outdir} 下现有 {len(got)} 个字幕文件。")
    for p in got[-8:]:
        print(f"   · {p.name}")
    if got:
        print("\n下一步入库：\n   ./update.sh")
    else:
        print("⚠ 没抓到字幕。可能该视频没有字幕，或换 --lang 试试（如 en / zh-Hans / zh-Hant）。")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="抓 YouTube 字幕 → video/课名/（供 ingest 入库）")
    ap.add_argument("urls", nargs="+", help="一个或多个视频/播放列表 URL")
    ap.add_argument("--course", required=True, help="课名（= video/ 下的文件夹名）")
    ap.add_argument("--series", default=None, help="系列名（可选，作中间层分组）")
    ap.add_argument("--lang", default="en,zh-Hans,zh-Hant",
                    help="字幕语言优先级，逗号分隔（默认 en,zh-Hans,zh-Hant）")
    ap.add_argument("--root", type=Path, default=Path("~/RAG-Database"),
                    help="RAG-Database 根目录")
    a = ap.parse_args()
    sys.exit(fetch(a.urls, a.course, a.root, a.series, a.lang))


if __name__ == "__main__":
    main()
