# Evaluating MedRAG

`medrag_eval.py` measures a **running** MedRAG backend end to end through its HTTP API: upload, OCR/parsing,
chunking, embeddings, retrieval, the similarity cutoff, the Gemini → Groq chain and the saved answers. It only needs
`httpx`:

```bash
pip install -r eval/requirements.txt
```

Every run signs in as a **throwaway account** and deletes it when it finishes, which removes everything it
uploaded and asked (`--keep` keeps it for inspection). To run as an existing account instead, pass `--email` and
`--password` (or set `MEDRAG_EMAIL` / `MEDRAG_PASSWORD`); then only the run's own documents and questions are deleted.
On an instance with `ALLOW_REGISTRATION=false`, create an evaluation account with `python -m app.cli create-user`
and use that. Reports go to `eval/results/` as Markdown and JSON.

**Use a separate database for large runs**, so your own documents and history are untouched and the app's document
list doesn't fill up with 900 abstracts while it runs:

```bash
docker compose exec db createdb -U postgres medrag_eval
docker compose run --rm -d --name medrag-eval -p 8002:8000 \
  -e DATABASE_URL=postgresql+psycopg://postgres:postgres@db:5432/medrag_eval backend
# ... run the evaluations with --base-url http://localhost:8002 ...
docker rm -f medrag-eval
```

## 1. Smoke set (2 minutes)

```bash
python eval/medrag_eval.py smoke --base-url http://localhost:8002
```

`datasets/smoke/`: five short documents in every supported format (two multi-page PDFs, a DOCX, two TXT files), with
deliberately confusable topics (asthma vs COPD, type 1 vs type 2 diabetes), and 33 questions:

- 25 answerable, each with the gold document, the gold PDF page and the facts the answer must contain;
- 5 medical questions the documents don't cover, and 3 off-topic ones, which must be refused.

It checks the plumbing and catches regressions. It is too small and too easy to say how good the system is: the
documents were written for it and retrieval is near-perfect.

## 2. PubMedQA (about 15 minutes)

```bash
python eval/medrag_eval.py pubmedqa --base-url http://localhost:8002 --answer-sample 60 --pause 7
```

[PubMedQA-L](https://pubmedqa.github.io/) is 1,000 research questions written from PubMed article titles, each with
its abstract and a yes / no / maybe answer **labelled by annotators with biomedical training** (MIT licence; downloaded
to `eval/.cache/` on first use). The harness:

1. Uploads 900 abstracts as documents (sections and conclusion, **not the title**, since the questions are written
   from the titles) and holds 100 back. Questions about held-back abstracts have no answer in the corpus, which makes
   realistic in-domain "should refuse" cases.
2. Runs retrieval for all 1,000 questions through `POST /api/search` (no LLM calls): is the question's own abstract
   in the top 1 / 3 / 5 / 10 among 900 biomedical abstracts?
3. Sweeps the similarity cutoff: how many answerable questions each value would wrongly refuse, and how many held-out
   ones it would catch before an LLM call.
4. Sends a sample (`--answer-sample`, default 60) through the full pipeline, asking the model to start with yes, no
   or maybe, and compares that with the expert label. Use `--pause` to stay inside free-tier rate limits.

`--n 200 --answer-sample 0` gives a quick retrieval-only run.

## 3. Clinician review

Automatic scores can't tell whether an answer is clinically right, complete or safe. To have clinicians grade answers:

```bash
python eval/medrag_eval.py review-export eval/results/<run>.json -o review.csv
```

The CSV (opens in Excel) has the question, the reference answer (the expert label and the abstract's conclusion for
PubMedQA), the system's answer, the model and the sources, plus columns for the reviewer to fill in:
`answer_correct` (yes / partly / no), `supported_by_sources` (yes / no), `potentially_harmful` (yes / no),
`reviewer` and `notes`. Give each reviewer their own copy, then:

```bash
python eval/medrag_eval.py review-summary review-dr-a.csv review-dr-b.csv
```

prints the grades per sheet, lists every answer marked potentially harmful, and, for two sheets, the reviewers'
agreement and Cohen's kappa. Low agreement means the grading guidance needs tightening before the numbers mean much.

Suggested protocol: at least two clinicians independently grade the same 50–100 answers drawn from your own
documents (PubMedQA abstracts are research findings, not the guidelines and handouts the app is meant for), and
anything marked harmful is reviewed together.

## Results

See `results/` for full reports. Baseline on 2026-10-04 (all-MiniLM-L6-v2 embeddings, chunk size 512, top_k 4,
Gemini `gemini-3.8-flash` → Groq `openai/gpt-oss-120b`):

| | Smoke set | PubMedQA |
|---|---|---|
| Documents / questions | 5 / 33 | 900 abstracts (+100 held out) / 1,000 |
| Right source ranked first | 100% (25/25) | 98.0% (882/900) |
| Right source in top 4 (what the LLM sees) | 100% (25/25) | 99.4% (895/900) |
| Answers correct | 100% (25/25, every key fact present) | 76.9% (40/52) agree with the expert yes/no/maybe label |
| Answerable questions refused | 0% (0/25) | 3.7% (2/54) |
| Unanswerable questions refused | 100% (8/8) | 83.3% (5/6) held-out questions |
| Refused by the cutoff, no LLM call | 6 of 33 | 1 of 100 held-out; 2 of 900 answerable |
| Median latency | 1.0 s | 1.5 s |

For context, the PubMedQA paper reports 78% accuracy for a single human annotator on this task. The LLM answers
come from a sample of 60, so treat 76.9% as roughly ±11 points.

What the failures show:

- **Retrieval misses** are rare: 5 of 900 questions had their abstract outside the top 4. One of the two refused
  answerable questions was such a miss (its abstract ranked 7th).
- **Over-caution**: the other refused question had its abstract ranked first; the model still refused.
- **Neighbour answers are the real risk.** One held-out question was answered "Maybe" from a *different* abstract
  on a related topic. The answer says the source does not state it directly, but a reader could still take it as
  the documents' answer. The similarity cutoff can't catch this (see below); clinician review and the visible
  sources are the safeguard.
- **"Maybe" is over-used**: 9 of the 12 disagreements are the model saying "maybe" where the expert said yes or no.

## Choosing `MIN_SIMILARITY`

The cutoff refuses a question without calling an LLM when even the best passage is a poor match. It saves quota and
latency, but every answerable question it refuses is a real failure, while the LLM itself refused every unanswerable
question in both sets. So the default is set **below the lowest answerable score seen**, not at the value that best
separates the two groups:

| Cutoff | Smoke: answerable refused | Smoke: unanswerable caught | PubMedQA: answerable refused | PubMedQA: held-out caught |
|---|---|---|---|---|
| 0.25 | 0/25 | 4/8 | 1/900 | 0/100 |
| 0.30 | 0/25 | 5/8 | 1/900 | 0/100 |
| **0.35 (default)** | **0/25** | **6/8** | **2/900** | **1/100** |
| 0.45 | 1/25 | 7/8 | 5/900 | 8/100 |
| 0.51 (best split on smoke) | 1/25 | 8/8 | 9/900 | 35/100 |
| 0.60 | 6/25 | 8/8 | 29/900 | 82/100 |

The cutoff is good at **off-topic** questions (capital of France, coding requests): they score far below any
medical passage. It is nearly useless for **in-domain** questions the documents don't answer: a held-out PubMedQA
question still finds a closely related abstract (median similarity 0.54 versus 0.81 when its own abstract is
there), so any cutoff that caught them would also refuse many answerable questions. Those are left to the LLM,
which refused 5 of 6 in the sample. The two answerable questions refused at 0.35 are metaphorical article titles
("…a disaster waiting to happen?", "Would a man smell a rose then throw it away?"), which real users rarely write.

Similarity values depend on the embedding model: re-run both sweeps and re-tune `MIN_SIMILARITY` if you change
`EMBEDDING_MODEL`, chunking, or the kind of documents you index.

## Limitations

- The yes/no/maybe agreement only checks the first word of the answer; the explanation isn't graded. That is what
  clinician review is for.
- PubMedQA abstracts are research findings. Clinical guidelines, handouts and notes may behave differently, so build
  a small question set from your own documents too (copy the smoke set's JSONL format).
- Free-tier quotas decide which model answers: with a 5-requests-per-minute Gemini key, most answers come from Groq.
