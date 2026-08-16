#!/usr/bin/env bash
# update.sh — 加了新资料后，一条命令更新知识库（分节 + 只嵌新增）。
#
# 用法：
#   ./update.sh                 # 用默认数据库目录
#   ./update.sh /别的/数据库路径   # 指定数据库目录
#
# 做两件事：
#   1) ingest：把 RAG-Database 重新切成 chunks.jsonl（书自动按页码分节；已有文件 id 不变）
#   2) buildindex：只嵌“新增/改动”的块（已嵌过的按 id 跳过），写进 vectors.npz
# 加 10 本书就只嵌那 10 本，不碰旧的几万块。几分钟、几分钱。
set -euo pipefail

SRC="${1:-~/RAG-Database}"
CHUNKS="chunks.jsonl"
VECTORS="vectors.npz"

if [ ! -d "$SRC" ]; then
  echo "✗ 找不到数据库目录：$SRC"
  echo "  用法：./update.sh /你的/RAG-Database路径"
  exit 1
fi

echo "▶ 1/2 ingest（分节）：$SRC → $CHUNKS"
python -m kb_rag.ingest --src "$SRC" --out "$CHUNKS"

echo
echo "▶ 2/2 buildindex（只嵌新增）：$CHUNKS → $VECTORS"
python -m kb_rag.buildindex --chunks "$CHUNKS" --out "$VECTORS"

echo
echo "✅ 更新完成。现在可以在本地 Claude Code 里直接问新加进来的内容了。"
