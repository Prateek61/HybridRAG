# Custom RAG

This is a from-scratch retrieval-augmented generation system for asking questions about a
collection of documents. It handles ingestion, passage retrieval, and answer generation, with the
retrieved passages supplied as evidence for each answer.

Retrieval quality is the main focus of the project. Vector search, keyword search, and reranking
are measured against the same held-out question set so their differences are visible. The stack
uses local tools and Groq's free API tier; no paid API is required.

## Stack

| Layer | Choice | Notes |
|---|---|---|
| Embeddings | `fastembed` (`BAAI/bge-base-en`, 768 dimensions) | Local ONNX, no PyTorch |
| Vector store | ChromaDB (`PersistentClient`) | Stored on disk |
| Keyword search | `rank_bm25` (`BM25Okapi`) | Lexical retrieval |
| Reranker | `fastembed` `TextCrossEncoder` (`BAAI/bge-reranker-base`) | Local ONNX cross-encoder |
| Generation and judging | Groq free API (`llama-3.1-8b-instant`) | Generates answers and scores them |
| API | FastAPI with a static chat UI | `/upload` and `/ask` endpoints |

## How a query is handled

```text
ingest → recursive chunking → embed → ChromaDB
                                          │
query ─► vector search ─┐                 │
        BM25 search   ──┼─► RRF fusion ─► cross-encoder rerank ─► top-k ─► prompt ─► LLM answer
                        ┘  (candidate_n=20)        (→ k)
```

Retrieval happens in two passes. Vector search and BM25 each produce candidates, and Reciprocal
Rank Fusion merges them into a pool of 20. The cross-encoder then scores each `(query, chunk)` pair
and returns the best `k` chunks.

## Evaluation

The numbers below come from `src/eval/run_eval.py` and `gold_questions.json`. The test set contains
59 questions: 55 answerable questions and 4 that are deliberately out of scope. The corpus has 91
chunks drawn from 9 fictional TechCorp documents and 3 archived policy documents.

`Recall@k` checks whether any of the first `k` chunks contains all relevant keywords after
whitespace normalization. `MRR` gives more credit when the correct chunk appears near the top.
The judge score asks an LLM to compare the generated answer with the expected answer on a
five-point scale.

### Why the corpus includes archived documents

The first version of the corpus was too easy. Every retriever came close to 1.0 recall@5, which hid
most differences in ranking.

The current corpus includes three archived policy documents that closely resemble the current
versions but contain old values. For example, the archived 401(k) policy says *50% up to 3%*, while
the current policy says *100% up to 4%*. Twelve questions ask specifically for a current value, and
only the current chunk counts as a hit.

That creates a practical ranking test. Keyword and dense retrieval can be pulled toward an archived
chunk because much of its wording still matches. The reranker has to use terms such as "current,"
"former," and "superseded" to put the right version first.

### Retrieval results

These results cover the 55 answerable questions in the 91-chunk corpus.

#### All answerable questions

| Metric | Vector only | Hybrid (RRF) | Hybrid + reranker |
|---|---:|---:|---:|
| Recall@1 | 0.764 | 0.727 | **0.818** |
| Recall@5 | 0.982 | 1.000 | 0.982 |
| MRR | 0.870 | 0.856 | **0.897** |

#### Temporal-disambiguation questions

| Metric | Vector only | Hybrid (RRF) | Hybrid + reranker |
|---|---:|---:|---:|
| Recall@1 | 0.417 | 0.333 | **0.583** |
| MRR | 0.708 | 0.653 | **0.792** |

### What the results showed

- Recall@5 has effectively topped out on this corpus. Recall@1 and MRR reveal the ranking
  differences much more clearly.
- Adding BM25 lowered hard-question Recall@1 from 0.417 to 0.333. The archived chunks share most of
  the same keywords as the current chunks, so lexical matching often promotes the wrong version.
- The `BAAI/bge-reranker-base` cross-encoder raised hard-question Recall@1 from 0.333 to 0.583. It
  also produced the best overall Recall@1 and MRR.
- The smaller `ms-marco-MiniLM-L-6-v2` reranker did not improve on hybrid retrieval for the hard
  questions; Recall@1 remained 0.333. The larger reranker earned its place through the evaluation.

### Why the judge score barely changes

Answer scores are close across the three retrievers: 4.16 for vector search, 4.04 for hybrid, and
4.13 for reranked retrieval. With `k=5`, all three usually place the correct chunk somewhere in the
context given to the generator. Better ordering therefore has little room to change the final
answer. For this experiment, Recall@1 and MRR say more about the reranker than the judge score at
`k=5`.

## Reproduce the evaluation

```bash
# install
./venv/bin/pip install -r requirements.txt

# ingest a document once
./venv/bin/python -m src.eval.run_eval --ingest path/to/file.pdf

# run the evaluation with each retrieval mode
./venv/bin/python -m src.eval.run_eval --retriever vector --label vector
./venv/bin/python -m src.eval.run_eval --retriever hybrid --label hybrid
./venv/bin/python -m src.eval.run_eval --retriever rerank --label rerank
```

Each run writes its results to `src/eval/results/<label>_<date>.json`.

## Run the app

```bash
./venv/bin/uvicorn src.main:app --reload
```

Open `http://localhost:8000` for the chat UI. You can also send a file to `/upload` and a question to
`/ask`.
