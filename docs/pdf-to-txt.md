# PDF → txt 处理流程（book 语料）

> 记录把书（PDF）转成检索用文本的完整流程与决策，供项目 review 查阅。

## 为什么这么做（背景决策）

1. **RAG 吃文本，不吃 PDF**。`ingest` 只读 `.txt/.md/.vtt/.srt`，不解析 PDF；检索/教学用的是纯文本。
2. **仓库只存文本，不存 PDF**。原因：
   - GitHub **网页上传单文件上限 25MB**，很多书超限；
   - 大 PDF 会**撑爆仓库**（clone 变慢、历史膨胀）；
   - PDF 只是"原料"，文本才是进库的东西。
3. **PDF 原件留在本地**，抽出的 `.txt` 才提交到语料仓库 `RAG-Database`，作为唯一真源，任何机器都能拉取。

链路：

```
本地 PDF（原料，留本地）
  → 抽取/OCR → .txt（本地生成）
  → 提交到 RAG-Database  book/<书名>/书.txt
  → RAG 拉仓库 → ingest → chunks.jsonl
```

## 两类 PDF，两条路

先判断 PDF 是哪一类（能否选中/复制其中文字）：

| 类型 | 特征 | 处理 |
|---|---|---|
| **文字版** | 能选中、复制文字 | 直接抽取，秒出 |
| **扫描版** | 整页是图片，选不中字；直接抽取得到**空文本** | 必须先 **OCR**（光学字符识别） |

> 判断捷径：抽出来是空的、或**只有极少几行**（如只剩一页 `SS号=…` 元数据）→ 就是扫描版，需 OCR。

## 工具

| 工具 | 用途 | 安装（macOS） |
|---|---|---|
| **PyMuPDF**（`fitz`） | 抽取文字版 PDF 的文本 | `pip install pymupdf` |
| **ocrmypdf** + **tesseract** | 给扫描版加文字层（OCR） | `brew install ocrmypdf tesseract-lang` |

`tesseract-lang` 含各语言包；OCR 语言：英文 `eng`，简体中文 `chi_sim`，繁体 `chi_tra`，中英混排 `chi_sim+eng`。**语言选错识别质量会很差。**

## 唯一工具：`kb_rag/pdf2txt.py`（全自动，自己识别）

一个脚本搞定，**不用你判断类型**。逻辑：

1. 先用 PyMuPDF 逐页抽文字层，统计"有字页比例 + 总字数"；
2. 自动判别：
   - 文字充足 → **文字版**，直接用（快、质量最好，不 OCR）；
   - 有字页比例 `< 0.6` **或** 全书总字数 `< 500` → **扫描/混合版**，调 `ocrmypdf --skip-text` OCR；
3. `--skip-text` 只 OCR 图片页、**保留已有文字页不动** → **混合版**（部分页有字、部分是图）也能只补缺字的页。

> 用比例+总字数判别，能兜住"只有一页元数据（如读秀 `SS号=…`）有文字层"的扫描书——那种抽出来非空但极短，用"文件是否为空"判断会漏掉。

每页正文前插 `[p.N]` 页码标记（便于引用到页/按页分节）。

```bash
python -m kb_rag.pdf2txt <书.pdf> [更多.pdf ...]     # 每个 → 同名 .txt，自动判别
python -m kb_rag.pdf2txt *.pdf                        # 批量
python -m kb_rag.pdf2txt 书.pdf -o "book/如何阅读一本书/书.txt"
python -m kb_rag.pdf2txt 书.pdf --lang chi_tra+eng    # 繁体
python -m kb_rag.pdf2txt 书.pdf --ocr force           # 强制整本 OCR
python -m kb_rag.pdf2txt 书.pdf --ocr off             # 只抽文字层、不 OCR
```

参数：`--lang`（OCR 语言，默认 `chi_sim+eng`）、`--ocr auto|force|off`、`--no-pages`、`-o`。

> 本地若没 clone 代码仓库：把 `kb_rag/pdf2txt.py` 内容存成 `~/pdf2txt.py`（它是自包含单文件），
> 用 `python ~/pdf2txt.py <PDF>` 即可。旧的 `book2txt` shell 函数已被本脚本取代，可弃用。

## 标准操作步骤

```bash
python -m kb_rag.pdf2txt "/path/to/书.pdf"      # 文字版/扫描版/混合版都用这一条
```

得到 `.txt` 后 → 放进 `RAG-Database` 的 `book/<书名>/书.txt`（**一本书一个文件夹**）→ 提交。

## 已知问题 / 注意点

- **换行过碎**：PDF 排版换行会被原样保留，txt 里每几个字一断行。对检索无大碍（分词忽略换行），但影响阅读与分节 → 计划在 book 分节时加"合并断行"。
- **OCR 质量**：
  - `[tesseract] lots of diacritics — possibly poor OCR` 是质量警告，**不代表一定差**，以实际 txt 内容为准（翻几页看中文是否读得通）。
  - 输出 PDF 的 `No installed font has glyphs...` 是**渲染**警告：文字层仍**可搜索可复制**（我们只要文本，PDF 丢弃）→ **忽略**。
  - 扫描件 OCR 天花板有限；若能找到**正版 EPUB/文字版**，质量远好于扫描 OCR，优先用。
- **语言参数**必须匹配书的语种，否则识别很差。
- **页码标记 `[p.N]`** 保留在 txt 中，供后续 book 分节做页级引用。

## 待办

- `ingest` 对 `kind=book` 的**专门分节**：一本几百页的书不能"整篇一节"（现状），需按 `[p.N]` 页码/章节切分，并合并断行、保留页码进元数据以支持页级引用。
