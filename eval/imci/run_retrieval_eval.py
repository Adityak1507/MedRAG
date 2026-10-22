"""Retrieval evaluation for MedRAG on the WHO IMCI Handbook.

Each question in questions_imci.jsonl has a short answer phrase copied from the handbook.
A retrieved chunk is relevant when its normalised text contains that phrase, so grading is
automatic and reproducible.

Metrics (averaged over questions):
  Recall@k    - share of questions with at least one relevant chunk in the top k (hit rate)
  Precision@k - share of the top-k chunks that are relevant
  MRR         - mean reciprocal rank of the first relevant chunk (0 if none in the top 10)

Usage (from the repo root, with the backend requirements installed):
  python eval/imci/run_retrieval_eval.py --retrievers bm25 minilm hybrid:bm25+minilm
  python eval/imci/run_retrieval_eval.py --retrievers bm25 spacy:en_core_sci_md --chunks 512/128
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from app.rag.loaders import extract_pages  # noqa: E402

KS = (1, 4, 10)
STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "be", "with", "what",
    "which", "how", "why", "when", "does", "do", "should", "can", "after", "before", "at", "by", "it",
    "this", "that", "as", "from", "into", "about", "who", "whom", "has", "have", "had", "his", "her",
}


def normalise(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # re-join words hyphenated across PDF lines
    return re.sub(r"\s+", " ", text).lower()


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP_WORDS]


def build_chunks(pdf: Path, size: int, overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    # OCR is off so results don't depend on Tesseract being installed (12 of 173 pages are image-only)
    for page in extract_pages(pdf, ".pdf", ocr_enabled=False):
        chunks.extend(c for c in splitter.split_text(page.text.replace("\x00", "")) if c.strip())
    return chunks


# --- retrievers: each returns the full ranking of chunk indices for a query -----------------

class BM25:
    name = "bm25"

    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([tokenize(c) for c in chunks])

    def rank(self, query):
        return np.argsort(-self.bm25.get_scores(tokenize(query)), kind="stable")


class Dense:
    def __init__(self, chunks, name, encode):
        self.name = name
        self.encode = encode
        self.matrix = self._normalised(encode(chunks))

    @staticmethod
    def _normalised(m):
        m = np.asarray(m, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)

    def rank(self, query):
        q = self._normalised(self.encode([query]))[0]
        return np.argsort(-(self.matrix @ q), kind="stable")


def spacy_encoder(model):
    import spacy

    nlp = spacy.load(model)
    disabled = list(nlp.pipe_names)

    def encode(texts):
        with nlp.select_pipes(disable=disabled):
            return [doc.vector for doc in nlp.pipe(texts, batch_size=64)]

    return encode


def minilm_encoder(model="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model)
    return lambda texts: st.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)


class Hybrid:
    """Reciprocal rank fusion of two retrievers."""

    def __init__(self, a, b, k=60):
        self.a, self.b, self.k = a, b, k
        self.name = f"hybrid:{a.name}+{b.name}"

    def rank(self, query):
        scores = {}
        for retriever in (self.a, self.b):
            for position, idx in enumerate(retriever.rank(query)[:200]):
                scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (self.k + position + 1)
        return np.array(sorted(scores, key=scores.get, reverse=True))


def make_retriever(spec, chunks, cache):
    if spec in cache:
        return cache[spec]
    if spec == "bm25":
        r = BM25(chunks)
    elif spec == "minilm":
        r = Dense(chunks, "minilm", minilm_encoder())
    elif spec.startswith("spacy:"):
        r = Dense(chunks, spec, spacy_encoder(spec.split(":", 1)[1]))
    elif spec.startswith("hybrid:"):
        left, right = spec.split(":", 1)[1].split("+")
        left = "spacy:" + left if left.startswith("en_") else left
        right = "spacy:" + right if right.startswith("en_") else right
        r = Hybrid(make_retriever(left, chunks, cache), make_retriever(right, chunks, cache))
        r.name = spec
    else:
        raise SystemExit(f"Unknown retriever {spec}")
    cache[spec] = r
    return r


def evaluate(retriever, chunks_norm, questions):
    hits = {k: 0 for k in KS}
    precision = {k: 0.0 for k in KS}
    rr = 0.0
    per_question = []
    for q in questions:
        ranking = retriever.rank(q["question"])[: max(KS)]
        relevant = [q["answer_phrase"] in chunks_norm[i] for i in ranking]
        first = next((pos for pos, rel in enumerate(relevant, 1) if rel), None)
        for k in KS:
            hits[k] += any(relevant[:k])
            precision[k] += sum(relevant[:k]) / k
        rr += 1.0 / first if first else 0.0
        per_question.append({"id": q["id"], "first_relevant_rank": first})
    n = len(questions)
    return {
        **{f"recall@{k}": hits[k] / n for k in KS},
        **{f"precision@{k}": precision[k] / n for k in KS},
        "mrr@10": rr / n,
        "per_question": per_question,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, default=ROOT / "HBprelims.pdf")
    parser.add_argument("--questions", type=Path, default=Path(__file__).with_name("questions_imci.jsonl"))
    parser.add_argument("--chunks", nargs="+", default=["256/64", "512/128", "1000/200"],
                        help="chunk_size/chunk_overlap settings to compare")
    parser.add_argument("--retrievers", nargs="+", default=["bm25", "minilm", "hybrid:bm25+minilm"])
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    for q in questions:
        q["answer_phrase"] = normalise(q["answer_phrase"])

    results = []
    for setting in args.chunks:
        size, overlap = map(int, setting.split("/"))
        chunks = build_chunks(args.pdf, size, overlap)
        chunks_norm = [normalise(c) for c in chunks]
        unanswerable = [q["id"] for q in questions if not any(q["answer_phrase"] in c for c in chunks_norm)]
        if unanswerable:
            print(f"warning: chunks {setting}: answer phrase not found in any chunk for questions {unanswerable}")
        cache = {}
        for spec in args.retrievers:
            metrics = evaluate(make_retriever(spec, chunks, cache), chunks_norm, questions)
            results.append({"chunks": setting, "num_chunks": len(chunks), "retriever": spec, **metrics})

    header = "| Chunks | Retriever | Recall@1 | Recall@4 | Recall@10 | Precision@4 | MRR@10 |"
    print(header)
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['chunks']} ({r['num_chunks']}) | {r['retriever']} | {r['recall@1']:.2f} | {r['recall@4']:.2f} | "
              f"{r['recall@10']:.2f} | {r['precision@4']:.2f} | {r['mrr@10']:.2f} |")

    args.out.write_text(json.dumps({"questions": len(questions), "results": results}, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
