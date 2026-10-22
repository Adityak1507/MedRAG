# IMCI handbook: design-choice and bug-fix comparisons

Retrieval evaluation on the WHO IMCI Handbook (`HBprelims.pdf`, 173 pages; OCR off, so the 12 image-only pages are not indexed), plus before/after measurements of the notebook bug fixes. It complements the end-to-end evaluation in `../README.md` (PubMedQA and the smoke set, run with the real model and LLMs) by comparing **alternatives**: chunk sizes, general vs biomedical embeddings, keyword vs dense vs hybrid search, and the original vs fixed notebook code.

## Method

- **Question set** (`questions_imci.jsonl`): 42 clinical questions, worded differently from the handbook text. Each has a short **answer phrase** copied from the handbook.
- **Relevance**: a retrieved chunk counts as relevant if its text contains the answer phrase, after normalising case, whitespace, quote marks and words hyphenated across line breaks. Grading is automatic and needs no human judgement.
- **Metrics**:
  - **Recall@k**: share of questions with at least one relevant chunk in the top k
  - **Precision@k**: share of the top-k chunks that are relevant
  - **MRR@10**: mean of 1 / rank of the first relevant chunk
- **Precision ceiling**: most answer phrases occur in only one chunk (median 1), so even a perfect retriever scores Precision@4 of at most **0.44–0.48**. Compare precision values against that ceiling, not against 1.0.

## Important caveat: stand-in embedders

The system's embedding model, `all-MiniLM-L6-v2`, couldn't be downloaded where these numbers were produced (Hugging Face was blocked). The dense results below use **averaged word vectors** in its place:
- `en_core_web_md`: general-English word vectors
- `en_core_sci_md`: scispaCy biomedical word vectors

Averaged word vectors are much weaker than a sentence-embedding model, so the **dense and hybrid numbers are a lower bound, not MiniLM's performance**. To get the real figures, rerun with `--retrievers bm25 minilm hybrid:bm25+minilm` wherever Hugging Face is reachable. BM25 numbers don't depend on a model.

## Results: retrieval (`run_retrieval_eval.py` → `results_proxy.json`)

| Chunks (count) | Retriever | Recall@1 | Recall@4 | Recall@10 | Precision@4 | MRR@10 |
|---|---|---|---|---|---|---|
| 256/64 (1949) | BM25 | 0.26 | 0.57 | 0.69 | 0.18 | 0.39 |
| 256/64 (1949) | dense, general vectors | 0.02 | 0.12 | 0.24 | 0.03 | 0.08 |
| 256/64 (1949) | dense, biomedical vectors | 0.05 | 0.14 | 0.19 | 0.04 | 0.09 |
| 256/64 (1949) | hybrid BM25 + biomedical | 0.05 | 0.29 | 0.40 | 0.08 | 0.16 |
| **512/128 (1034)** | BM25 | 0.43 | 0.69 | 0.81 | 0.23 | 0.55 |
| **512/128 (1034)** | dense, general vectors | 0.10 | 0.21 | 0.36 | 0.05 | 0.15 |
| **512/128 (1034)** | dense, biomedical vectors | 0.12 | 0.36 | 0.48 | 0.09 | 0.23 |
| **512/128 (1034)** | hybrid BM25 + biomedical | 0.29 | 0.55 | 0.64 | 0.15 | 0.41 |
| 1000/200 (532) | BM25 | 0.52 | 0.74 | 0.88 | 0.24 | 0.63 |
| 1000/200 (532) | dense, general vectors | 0.05 | 0.14 | 0.33 | 0.04 | 0.12 |
| 1000/200 (532) | dense, biomedical vectors | 0.26 | 0.48 | 0.57 | 0.12 | 0.37 |
| 1000/200 (532) | hybrid BM25 + biomedical | 0.43 | 0.64 | 0.76 | 0.20 | 0.54 |

**512/128** is the web app's and the notebook demo's setting. 1000/200 is the notebook's original `RAGConfig` default.

## Results: notebook bug fixes (`notebook_bugfix_impact.py` → `notebook_bugfix_results.json`)

Both runs use the notebook demo settings (512/128, top-4) and biomedical vectors in place of MiniLM.

| Bug | Before fix | After fix |
|---|---|---|
| ChromaDB queries returning results (out of 42) | 0 (every query errored) | 42 |
| ChromaDB Recall@4 | 0.00 | 0.36 (same as FAISS) |
| FAISS with top_k larger than the chunk count (2-chunk document, top_k 5) | 5 results, 3 duplicates (2 of 5 unique = 40%) | 2 results, 0 duplicates (100%) |
| No-LLM fallback shows the answer (15 questions whose retrieved context contains it) | 4 / 15 (27%) | 8 / 15 (53%) |
| Share of context lines the fallback treats as relevant | 81% (barely filters) | 62% |

## Not yet evaluated

- **The real MiniLM model**, and **answer quality** (correctness, faithfulness, refusing unanswerable questions). Answer quality needs an LLM API key and either human grading or an LLM judge.
- **Bias**: the questions were written by reading the handbook, which can favour keyword matching (BM25) despite the rewording. A question set written independently by clinicians would be a stronger test.

## Reproduce

```bash
pip install -r backend/requirements.txt -r eval/imci/requirements.txt
python eval/imci/run_retrieval_eval.py --retrievers bm25 minilm hybrid:bm25+minilm   # real model
python eval/imci/run_retrieval_eval.py --retrievers bm25 spacy:en_core_web_md spacy:en_core_sci_md hybrid:bm25+en_core_sci_md
python eval/imci/notebook_bugfix_impact.py
```
