# kb-rag

A personal, **hallucination-resistant**, **cross-lingual** *agentic* RAG over your own study
materials — English course transcripts, Chinese books (PDF → text), video subtitles. Retrieval is
deterministic and local (essentially free); reasoning is done by an LLM only where it actually helps.

It runs in two modes:

- **Local & free** — use any Claude session as the reasoning agent on top of a pure-local retrieval
  CLI. Retrieval is deterministic computation on your machine, so there is **no API cost** for
  reasoning; the only (tiny) cost is embedding newly added chunks.
- **Always-on WhatsApp** — an optional cloud service (FastAPI + Twilio) that answers from the same
  knowledge base using a cheap model (OpenAI `gpt-4o-mini`), deployable to Railway.

---

## Why it exists

Three goals shaped every design decision:

1. **No hallucination.** Answers may only use retrieved source text. Every claim maps to a real
   retrieved chunk id, and a semantic **verify** gate (embedding cosine, not the model's word) blocks
   any answer that isn't actually supported — otherwise it refuses. Cited ids must be ones that
   retrieval really returned, so ids can't be fabricated.
2. **Cross-domain / cross-lingual.** Ask in Chinese, get answers grounded in English sources (and vice
   versa). Lexical search alone can't bridge languages, so retrieval is **hybrid** (lexical + vector).
3. **Genuinely agentic.** Not a hard-wired pipeline: the loop plans, searches, judges whether the
   evidence is sufficient, reformulates the query (including switching language), fetches full
   sections, runs the anti-hallucination check, then answers or refuses.

---

## How retrieval works

```
question
  ├── BM25 / TF-IDF lexical  (CJK bigram tokenizer, no external segmenter)
  └── vector embedding → cosine  (bge-m3 / e5 locally, or OpenAI / Voyage)
                 │
                 └── RRF fusion → (optional cross-encoder rerank) → top-k evidence
                                              │
                                              └── semantic verify gate (anti-hallucination)
```

- **Hybrid + RRF.** Reciprocal-rank fusion of a lexical and a vector ranking. Cross-lingual recall
  rides on the vector half (lexical coverage is 0 when the query and source share no words).
- **Optional rerank.** A cross-encoder (`bge-reranker-base`) re-scores a larger candidate pool and is
  the biggest lever for the hard "ask in Chinese about English content" cases.
- **Verify gate.** The final answer is cosine-checked against the cited source chunks; below a floor
  (`KB_RAG_SEM_FLOOR`, default 0.30) the answer is blocked. This is robust to paraphrase and to
  cross-language answering.
- **Page-accurate citations.** Books are sectioned by page window (`p.a-b`), so citations point to a
  page range you can open and check.

Pluggable embedders (`KB_RAG_EMBEDDER`): `local` (sentence-transformers, offline & free),
`openai` (`text-embedding-3-small`, supports Matryoshka down-projection via `KB_RAG_OPENAI_DIM`),
`voyage`, or `fake` (deterministic, for offline tests). **Query and corpus must use the same
embedder/dimension** — changing it means re-embedding.

---

## Quick start

```bash
# 1. Install (local, offline, free embeddings)
pip install -e ".[vector,local]"          # or ".[vector]" to use OpenAI embeddings

# 2. Ingest your materials into chunks (books auto-sectioned by page markers)
python -m kb_rag.ingest --src /path/to/RAG-Database --out chunks.jsonl

# 3. Build the vector index (incremental — only new chunks are embedded)
export KB_RAG_EMBEDDER=local               # or openai (needs OPENAI_API_KEY)
python -m kb_rag.buildindex --chunks chunks.jsonl --out vectors.npz

# 4. Retrieve — pure local, no LLM, no API cost
python -m kb_rag.retrieve search "what is the anchoring effect" --k 8
python -m kb_rag.retrieve outline "Thinking, Fast and Slow"
python -m kb_rag.retrieve section "Thinking, Fast and Slow" "p.131-138"
python -m kb_rag.retrieve verify "your drafted answer" <chunk_id> <chunk_id> ...
```

Data layout (bucket name sets the `kind`, subfolder name becomes the cited source title):

```
RAG-Database/
  book/<Title>/<Title>.txt          kind=book   (books sectioned by [p.N] page markers)
  course/[series/]<Course>/*.txt    kind=course
  video/<Course>/*.vtt|*.txt        kind=video
```

Use `kb-rag`'s own reasoning agent instead of the pure CLI:

```bash
python -m kb_rag.cli ask "how does human-in-the-loop approval work" --agentic --llm openai
```

---

## The two modes

| | Local (free) | Cloud WhatsApp |
|---|---|---|
| Runs on | your machine | Railway (24/7) |
| Reasoning | any Claude session, via the `retrieve` CLI | OpenAI `gpt-4o-mini` |
| Embeddings | local bge-m3 / e5 (offline) | OpenAI (no torch in the container) |
| Cost | $0 API for reasoning | a fraction of a cent per question |

The cloud path ships without `torch`, so it uses OpenAI query embeddings; the corpus vectors are
down-projected and stored in float16 (`kb_rag/shrinkvecs.py`) to keep the image small. See
[`docs/deploy-railway.md`](docs/deploy-railway.md) for the full Railway walkthrough, and
[`docs/whatsapp-railway-journal.md`](docs/whatsapp-railway-journal.md) for a blow-by-blow of the
pitfalls hit while deploying.

---

## Architecture & evaluation

- [`docs/architecture.html`](docs/architecture.html) — visual system diagram (runtime path, retrieval
  internals, data pipeline, the two backends).
- [`docs/retrieval-scores.md`](docs/retrieval-scores.md) — a reproducible golden-set evaluation
  (Recall@k / MRR by language and material type), tracked across changes.

> Note: some in-repo docs and code comments are written in Chinese, reflecting the bilingual corpus
> this was built for.

---

## Project layout

```
kb_rag/
  ingest.py      raw materials → chunks.jsonl (metadata; page-based book sectioning)
  embed.py       pluggable embedders (local / openai / voyage / fake)
  buildindex.py  incremental vector index (only embeds new chunks)
  vecstore.py    content-addressed vector store (.npz)
  engine.py      BM25 + TF-IDF lexical retrieval (CJK bigram)
  hybrid.py      hybrid retrieval: lexical + vector, RRF fusion, optional rerank
  rerank.py      cross-encoder reranker
  retrieve.py    pure-local retrieval CLI (no LLM) — search / section / outline / verify
  agent.py       reasoning agent (plan → retrieve → synthesize → self-check)
  agentic.py     native tool-use loops (OpenAI / Anthropic)
  server.py      WhatsApp (Twilio) frontend
  shrinkvecs.py  down-project + float16 compress vectors for cloud deploy
  eval.py        golden-set retrieval evaluation (Recall@k / MRR)
```

---

## License

[MIT](LICENSE).
