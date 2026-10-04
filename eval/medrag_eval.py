"""Evaluate a running MedRAG instance through its HTTP API.

    python eval/medrag_eval.py smoke     [--base-url URL]           # 33 hand-written questions, 5 documents
    python eval/medrag_eval.py pubmedqa  [--base-url URL] [--n 1000] # 1,000 expert-labelled PubMedQA questions
    python eval/medrag_eval.py review-export  RESULTS.json [-o review.csv]  # sheet for clinician grading
    python eval/medrag_eval.py review-summary review.csv [more.csv ...]    # summarise completed sheets

Runs sign in as a throwaway account that is deleted afterwards, with everything it uploaded (or as an existing
account with --email/--password or MEDRAG_EMAIL/MEDRAG_PASSWORD; then only the run's own documents and questions
are deleted). --keep keeps everything for inspection. See eval/README.md.
"""

import argparse
import csv
import json
import os
import random
import secrets
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import httpx

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
CACHE_DIR = HERE / ".cache"
PUBMEDQA_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
CUTOFF_SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


# --------------------------------------------------------------------------- API client


class MedRAG:
    def __init__(self, base_url: str, email: str | None = None, password: str | None = None):
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=180)
        self.doc_ids: list[int] = []
        self.query_ids: list[int] = []
        self.throwaway = email is None
        if self.throwaway:
            email, password = f"eval-{secrets.token_hex(6)}@example.org", secrets.token_urlsafe(18)
            resp = self.http.post("/api/auth/register", json={"email": email, "password": password, "name": "Evaluation"})
        else:
            resp = self.http.post("/api/auth/login", json={"email": email, "password": password})
        if resp.status_code >= 400:
            sys.exit(f"Could not sign in as {email}: {resp.status_code} {resp.text}")
        self.password = password
        # Bearer token rather than the cookie, which is scoped to /api and set for the browser
        self.http.headers["Authorization"] = f"Bearer {resp.json()['token']}"
        print(f"Signed in as {email}" + (" (throwaway account)" if self.throwaway else ""))

    def _call(self, method: str, path: str, **kw) -> httpx.Response:
        for attempt in range(5):
            resp = self.http.request(method, path, **kw)
            if resp.status_code not in (429, 502, 503, 504):
                resp.raise_for_status()
                return resp
            time.sleep(2 * (attempt + 1))
        resp.raise_for_status()
        return resp

    def info(self) -> dict:
        return self._call("GET", "/api/info").json()

    def upload(self, name: str, data: bytes) -> int:
        doc = self._call("POST", "/api/documents", files={"file": (name, data)}).json()
        self.doc_ids.append(doc["id"])
        return doc["id"]

    def wait_ready(self, ids: list[int], timeout: float = 1800) -> dict[int, dict]:
        """Poll until none of `ids` is processing; return their final state."""
        wanted, deadline = set(ids), time.time() + timeout
        while True:
            docs, offset = {}, 0
            while True:
                resp = self._call("GET", "/api/documents", params={"limit": 500, "offset": offset})
                page = resp.json()
                docs.update({d["id"]: d for d in page if d["id"] in wanted})
                offset += len(page)
                if not page or offset >= int(resp.headers.get("x-total-count", offset)):
                    break
            pending = [i for i in wanted if docs.get(i, {}).get("status", "processing") == "processing"]
            if not pending:
                return docs
            if time.time() > deadline:
                raise TimeoutError(f"{len(pending)} documents still processing")
            print(f"  waiting for {len(pending)} documents to finish processing...", flush=True)
            time.sleep(5)

    def search(self, question: str, top_k: int, doc_ids: list[int]) -> list[dict]:
        body = {"question": question, "top_k": top_k, "document_ids": doc_ids}
        return self._call("POST", "/api/search", json=body).json()

    def ask(self, question: str, top_k: int, doc_ids: list[int]) -> tuple[dict, float]:
        start = time.perf_counter()
        body = {"question": question, "top_k": top_k, "document_ids": doc_ids}
        result = self._call("POST", "/api/query", json=body).json()
        self.query_ids.append(result["id"])
        return result, time.perf_counter() - start

    def cleanup(self) -> None:
        if self.throwaway:
            self.http.request("DELETE", "/api/auth/me", json={"password": self.password}).raise_for_status()
            print(f"Deleted the throwaway account with its {len(self.doc_ids)} documents and {len(self.query_ids)} questions.")
            return
        for qid in self.query_ids:
            self.http.delete(f"/api/queries/{qid}")
        for did in self.doc_ids:
            self.http.delete(f"/api/documents/{did}")
        print(f"Cleaned up {len(self.doc_ids)} documents and {len(self.query_ids)} saved questions.")


# --------------------------------------------------------------------------- scoring helpers


def normalise(text: str) -> str:
    """Lower-case, with hyphens (ASCII or unicode) and runs of whitespace treated alike."""
    return re.sub(r"[\s\-‐-―−]+", " ", text.lower())


def mentions(text: str, phrase: str) -> bool:
    """Whole-word match, case-insensitive, allowing a plural 's'."""
    return re.search(r"(?<![a-z0-9])" + re.escape(normalise(phrase)) + r"s?(?![a-z0-9])", normalise(text)) is not None


def refused(answer: str) -> bool:
    return "cannot answer" in answer.lower()


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}% ({n}/{d})" if d else "n/a"


def latency_summary(values: list[float]) -> dict:
    values = sorted(values)
    return {
        "p50": round(statistics.median(values), 2),
        "p95": round(values[int(0.95 * (len(values) - 1))], 2),
        "max": round(values[-1], 2),
    }


def cutoff_table(answerable: list[float], unanswerable: list[float]) -> list[dict]:
    """For each cutoff: answerable questions it would wrongly refuse, unanswerable ones it would catch."""
    return [
        {
            "cutoff": t,
            "false_refusals": sum(s < t for s in answerable),
            "answerable": len(answerable),
            "caught": sum(s < t for s in unanswerable),
            "unanswerable": len(unanswerable),
        }
        for t in CUTOFF_SWEEP
    ]


def save(name: str, results: dict, report: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{name}"
    stem.with_suffix(".json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    stem.with_suffix(".md").write_text(report, encoding="utf-8")
    print(f"\n{report}\nSaved {stem}.json and .md")
    return stem


def cutoff_markdown(table: list[dict], configured: float) -> list[str]:
    lines = ["| Cutoff | Answerable wrongly refused | Unanswerable caught before the LLM |", "|---|---|---|"]
    for row in table:
        mark = " (configured)" if abs(row["cutoff"] - configured) < 1e-9 else ""
        lines.append(
            f"| {row['cutoff']:.2f}{mark} | {pct(row['false_refusals'], row['answerable'])} | "
            f"{pct(row['caught'], row['unanswerable'])} |"
        )
    return lines


# --------------------------------------------------------------------------- smoke set


def run_smoke(args) -> None:
    api = MedRAG(args.base_url, args.email, args.password)
    info = api.info()
    data = HERE / "datasets" / "smoke"
    questions = [json.loads(line) for line in (data / "questions.jsonl").read_text(encoding="utf-8").splitlines()]

    name_to_id = {}
    try:
        for path in sorted((data / "corpus").iterdir()):
            name_to_id[path.name] = api.upload(path.name, path.read_bytes())
        docs = api.wait_ready(list(name_to_id.values()))
        failed = [d["filename"] for d in docs.values() if d["status"] != "ready"]
        if failed:
            sys.exit(f"Documents failed to process: {failed}")
        ids = list(name_to_id.values())

        rows = []
        for q in questions:
            result, seconds = api.ask(q["question"], args.top_k, ids)
            files = [s["filename"] for s in result["sources"]]
            row = {
                **q,
                "answer": result["answer"],
                "llm": result["llm"],
                "latency": round(seconds, 2),
                "top_similarity": result["sources"][0]["similarity"] if result["sources"] else None,
                "sources": [(s["filename"], s["page"], s["similarity"]) for s in result["sources"]],
                "refused": refused(result["answer"]),
            }
            if q["answerable"]:
                row["rank"] = files.index(q["gold_file"]) + 1 if q["gold_file"] in files else None
                if q.get("gold_page"):
                    row["page_hit"] = any(f == q["gold_file"] and p == q["gold_page"] for f, p, _ in row["sources"])
                row["correct"] = all(any(mentions(result["answer"], k) for k in group) for group in q["must_include"])
            rows.append(row)
            print(f"{q['id']:4} {seconds:5.1f}s {result['llm'][:34]:34} "
                  + (f"rank={row['rank']} correct={row['correct']}" if q["answerable"] else f"refused={row['refused']}"),
                  flush=True)
            time.sleep(args.pause)
    finally:
        if not args.keep:
            api.cleanup()

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    pages = [r for r in ans if "page_hit" in r]
    llm_calls = sum(r["llm"] not in ("similarity_cutoff", "retrieval_only") for r in rows)
    summary = {
        "hit_at_1": sum(r["rank"] == 1 for r in ans) / len(ans),
        "hit_at_k": sum(r["rank"] is not None for r in ans) / len(ans),
        "mrr": statistics.mean(1 / r["rank"] if r["rank"] else 0 for r in ans),
        "page_hit": sum(r["page_hit"] for r in pages) / len(pages),
        "answer_correct": sum(r["correct"] for r in ans) / len(ans),
        "false_refusals": sum(r["refused"] for r in ans) / len(ans),
        "unanswerable_refused": sum(r["refused"] for r in una) / len(una),
        "refused_by_cutoff": sum(r["llm"] == "similarity_cutoff" for r in rows),
        "llm_calls": llm_calls,
        "answered_by": Counter(r["llm"] for r in rows),
        "latency": latency_summary([r["latency"] for r in rows]),
    }
    table = cutoff_table([r["top_similarity"] for r in ans], [r["top_similarity"] or 0 for r in una])
    wrong = [r for r in ans if not r["correct"]]

    report = "\n".join([
        f"# Smoke evaluation ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"Instance: {args.base_url} · LLM chain: {info['llm']} · top_k={args.top_k} · "
        f"min_similarity={info.get('min_similarity')}",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Gold document ranked first | {pct(sum(r['rank'] == 1 for r in ans), len(ans))} |",
        f"| Gold document in sources | {pct(sum(r['rank'] is not None for r in ans), len(ans))} |",
        f"| Gold PDF page in sources | {pct(sum(r['page_hit'] for r in pages), len(pages))} |",
        f"| Answers containing every key fact | {pct(sum(r['correct'] for r in ans), len(ans))} |",
        f"| Answerable questions refused | {pct(sum(r['refused'] for r in ans), len(ans))} |",
        f"| Unanswerable questions refused | {pct(sum(r['refused'] for r in una), len(una))} |",
        f"| Refused by the similarity cutoff (no LLM call) | {summary['refused_by_cutoff']} of {len(rows)} |",
        f"| Latency p50 / p95 / max | {summary['latency']['p50']}s / {summary['latency']['p95']}s / "
        f"{summary['latency']['max']}s |",
        "",
        "Answered by: " + ", ".join(f"{k} ({v})" for k, v in summary["answered_by"].most_common()),
        "",
        "## Similarity cutoff sweep",
        "",
        *cutoff_markdown(table, info.get("min_similarity", 0)),
        "",
        "## Answers missing a key fact (check by hand: the grader is a keyword match)",
        "",
        *([f"- **{r['id']}** {r['question']}\n  > {r['answer'][:300]}" for r in wrong] or ["None."]),
        "",
    ])
    save("smoke", {"summary": summary, "cutoff_sweep": table, "answers": rows}, report)


# --------------------------------------------------------------------------- PubMedQA


def load_pubmedqa() -> dict:
    path = CACHE_DIR / "ori_pqal.json"
    if not path.exists():
        print(f"Downloading PubMedQA-L (MIT licence) from {PUBMEDQA_URL}")
        CACHE_DIR.mkdir(exist_ok=True)
        path.write_bytes(httpx.get(PUBMEDQA_URL, follow_redirects=True, timeout=120).raise_for_status().content)
    return json.loads(path.read_text(encoding="utf-8"))


def abstract_text(item: dict) -> str:
    """The abstract as a user would upload it: labelled sections plus the conclusion, but not the title
    (PubMedQA questions are written from the titles, so including them would make retrieval trivial)."""
    sections = [f"{label}: {text}" for label, text in zip(item["LABELS"], item["CONTEXTS"])]
    return "\n\n".join(sections + [f"CONCLUSIONS: {item['LONG_ANSWER']}"])


def yes_no_maybe(answer: str) -> str:
    if refused(answer):
        return "refused"
    first = re.sub(r"[^a-z]", "", normalise(answer).strip().split()[0]) if answer.strip() else ""
    return first if first in ("yes", "no", "maybe") else "unclear"


def run_pubmedqa(args) -> None:
    data = load_pubmedqa()
    rng = random.Random(args.seed)
    pmids = sorted(data)
    if args.n < len(pmids):
        pmids = sorted(rng.sample(pmids, args.n))
    held_out = set(rng.sample(pmids, round(len(pmids) * args.holdout)))
    indexed = [p for p in pmids if p not in held_out]
    print(f"{len(indexed)} abstracts indexed, {len(held_out)} held out (their questions should be refused)")

    api = MedRAG(args.base_url, args.email, args.password)
    info = api.info()
    try:
        pmid_to_doc = {}
        for i, pmid in enumerate(indexed, 1):
            pmid_to_doc[pmid] = api.upload(f"pubmedqa_{pmid}.txt", abstract_text(data[pmid]).encode())
            if i % 100 == 0:
                print(f"  uploaded {i}/{len(indexed)}", flush=True)
        docs = api.wait_ready(list(pmid_to_doc.values()))
        doc_ids = [d for d in pmid_to_doc.values() if docs[d]["status"] == "ready"]
        doc_to_pmid = {d: p for p, d in pmid_to_doc.items()}

        # Retrieval for every question (no LLM calls)
        retrieval = []
        for i, pmid in enumerate(pmids, 1):
            hits = api.search(data[pmid]["QUESTION"], 10, doc_ids)
            ranked = [doc_to_pmid[h["document_id"]] for h in hits]
            # Several chunks can come from one abstract; rank by first appearance
            ranked = list(dict.fromkeys(ranked))
            retrieval.append({
                "pmid": pmid,
                "answerable": pmid not in held_out,
                "rank": ranked.index(pmid) + 1 if pmid in ranked else None,
                "top_similarity": hits[0]["similarity"] if hits else 0.0,
            })
            if i % 200 == 0:
                print(f"  searched {i}/{len(pmids)}", flush=True)

        # LLM answers for a sample, graded against the expert yes/no/maybe label
        answers = []
        if args.answer_sample:
            n_out = min(len(held_out), max(1, round(args.answer_sample * args.holdout)))
            sample = rng.sample(indexed, args.answer_sample - n_out) + rng.sample(sorted(held_out), n_out)
            for i, pmid in enumerate(sample, 1):
                question = data[pmid]["QUESTION"] + "\nStart your answer with Yes, No or Maybe, then explain briefly."
                result, seconds = api.ask(question, args.top_k, doc_ids)
                answers.append({
                    "pmid": pmid,
                    "question": data[pmid]["QUESTION"],
                    "answerable": pmid not in held_out,
                    "expert_label": data[pmid]["final_decision"],
                    "expert_long_answer": data[pmid]["LONG_ANSWER"],
                    "answer": result["answer"],
                    "system_label": yes_no_maybe(result["answer"]),
                    "llm": result["llm"],
                    "latency": round(seconds, 2),
                    "sources": [(s["filename"], s["page"], s["similarity"]) for s in result["sources"]],
                })
                print(f"  answer {i}/{len(sample)} {seconds:5.1f}s {result['llm'][:30]:30} "
                      f"expert={data[pmid]['final_decision']:5} system={answers[-1]['system_label']}", flush=True)
                time.sleep(args.pause)
    finally:
        if not args.keep:
            api.cleanup()

    ans = [r for r in retrieval if r["answerable"]]
    una = [r for r in retrieval if not r["answerable"]]
    hit = lambda k: sum(r["rank"] is not None and r["rank"] <= k for r in ans)
    table = cutoff_table([r["top_similarity"] for r in ans], [r["top_similarity"] for r in una])
    summary = {
        "indexed": len(indexed),
        "held_out": len(held_out),
        "hit_at": {k: hit(k) / len(ans) for k in (1, 3, 5, 10)},
        "mrr_at_10": statistics.mean(1 / r["rank"] if r["rank"] else 0 for r in ans),
        "similarity_answerable_median": statistics.median(r["top_similarity"] for r in ans),
        "similarity_unanswerable_median": statistics.median(r["top_similarity"] for r in una) if una else None,
    }

    lines = [
        f"# PubMedQA evaluation ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"Instance: {args.base_url} · LLM chain: {info['llm']} · embeddings: {info['embedding_model']} · "
        f"min_similarity={info.get('min_similarity')}",
        "",
        f"Corpus: {len(indexed)} PubMed abstracts from PubMedQA-L (expert-labelled), {len(held_out)} more held out "
        "so that their questions have no answer in the corpus. Seed "
        f"{args.seed}.",
        "",
        "## Retrieval (all questions, no LLM)",
        "",
        "| Metric | Result |",
        "|---|---|",
        *[f"| Source abstract in top {k} | {pct(hit(k), len(ans))} |" for k in (1, 3, 5, 10)],
        f"| MRR@10 | {summary['mrr_at_10']:.3f} |",
        f"| Median top similarity, answerable / held out | {summary['similarity_answerable_median']:.3f} / "
        + (f"{summary['similarity_unanswerable_median']:.3f} |" if una else "n/a |"),
        "",
        "## Similarity cutoff sweep",
        "",
        *cutoff_markdown(table, info.get("min_similarity", 0)),
        "",
    ]
    if answers:
        a_in = [r for r in answers if r["answerable"]]
        a_out = [r for r in answers if not r["answerable"]]
        decided = [r for r in a_in if r["system_label"] in ("yes", "no", "maybe")]
        quota = [r for r in answers if r["llm"] == "all LLMs failed"]
        confusion = Counter((r["expert_label"], r["system_label"]) for r in a_in)
        summary["answers"] = {
            "agreement_with_expert": sum(r["system_label"] == r["expert_label"] for r in decided) / max(1, len(decided)),
            "answerable_refused": sum(r["system_label"] == "refused" for r in a_in) / max(1, len(a_in)),
            "held_out_refused": sum(r["system_label"] == "refused" for r in a_out) / max(1, len(a_out)),
            "answered_by": Counter(r["llm"] for r in answers),
            "latency": latency_summary([r["latency"] for r in answers]),
        }
        labels = ["yes", "no", "maybe", "refused", "unclear"]
        lines += [
            f"## LLM answers (sample of {len(answers)}: {len(a_in)} answerable, {len(a_out)} held out)",
            "",
            "| Metric | Result |",
            "|---|---|",
            f"| Yes/no/maybe agrees with the expert label (answerable, when a label was given) | "
            f"{pct(sum(r['system_label'] == r['expert_label'] for r in decided), len(decided))} |",
            f"| Answerable questions refused | {pct(sum(r['system_label'] == 'refused' for r in a_in), len(a_in))} |",
            f"| Answerable questions with no clear yes/no/maybe | "
            f"{pct(sum(r['system_label'] == 'unclear' for r in a_in), len(a_in))} |",
            f"| Held-out questions refused | {pct(sum(r['system_label'] == 'refused' for r in a_out), len(a_out))} |",
            f"| Both LLMs failed (quota/outage) | {len(quota)} |",
            f"| Latency p50 / p95 | {summary['answers']['latency']['p50']}s / {summary['answers']['latency']['p95']}s |",
            "",
            "Answered by: " + ", ".join(f"{k} ({v})" for k, v in summary["answers"]["answered_by"].most_common()),
            "",
            "Expert label (rows) vs system answer (columns), answerable questions:",
            "",
            "| | " + " | ".join(labels) + " |",
            "|---" * (len(labels) + 1) + "|",
            *[f"| **{e}** | " + " | ".join(str(confusion[(e, s)]) for s in labels) + " |" for e in ("yes", "no", "maybe")],
            "",
        ]
    save("pubmedqa", {"summary": summary, "cutoff_sweep": table, "retrieval": retrieval, "answers": answers},
         "\n".join(lines))


# --------------------------------------------------------------------------- clinician review

REVIEW_COLUMNS = ["answer_correct (yes/partly/no)", "supported_by_sources (yes/no)",
                  "potentially_harmful (yes/no)", "reviewer", "notes"]


def review_export(args) -> None:
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    out = Path(args.output or Path(args.results).with_suffix(".review.csv"))
    with out.open("w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel opens it as UTF-8
        writer = csv.writer(f)
        writer.writerow(["id", "question", "reference_answer", "system_answer", "model", "sources", *REVIEW_COLUMNS])
        for r in results["answers"]:
            reference = r.get("expert_long_answer") or ""
            if r.get("expert_label"):
                reference = f"[{r['expert_label']}] {reference}"
            sources = "\n".join(f"{name} p{page} (sim {sim:.2f})" if page else f"{name} (sim {sim:.2f})"
                                for name, page, sim in r["sources"])
            writer.writerow([r.get("id") or r.get("pmid"), r["question"], reference, r["answer"], r["llm"], sources,
                             *[""] * len(REVIEW_COLUMNS)])
    print(f"Wrote {len(results['answers'])} rows to {out}. Reviewers fill in the last {len(REVIEW_COLUMNS)} columns.")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def review_summary(args) -> None:
    sheets = []
    for path in args.sheets:
        with open(path, newline="", encoding="utf-8-sig") as f:
            sheets.append({row["id"]: row for row in csv.DictReader(f)})
    col = lambda name: next(c for c in REVIEW_COLUMNS if c.startswith(name))
    for path, sheet in zip(args.sheets, sheets):
        graded = [r for r in sheet.values() if r[col("answer_correct")].strip()]
        values = lambda name: Counter(r[col(name)].strip().lower() for r in graded)
        print(f"\n{path}: {len(graded)} of {len(sheet)} rows graded")
        for name in ("answer_correct", "supported_by_sources", "potentially_harmful"):
            print(f"  {name:22} " + ", ".join(f"{k}: {pct(v, len(graded))}" for k, v in values(name).most_common()))
        harmful = [r for r in graded if r[col("potentially_harmful")].strip().lower() == "yes"]
        for r in harmful:
            print(f"  ! potentially harmful: {r['id']} {r['question'][:80]} | {r[col('notes')][:120]}")
    if len(sheets) == 2:
        shared = [i for i in sheets[0] if i in sheets[1]
                  and sheets[0][i][col("answer_correct")].strip() and sheets[1][i][col("answer_correct")].strip()]
        if shared:
            a = [sheets[0][i][col("answer_correct")].strip().lower() for i in shared]
            b = [sheets[1][i][col("answer_correct")].strip().lower() for i in shared]
            agree = sum(x == y for x, y in zip(a, b))
            print(f"\nReviewer agreement on answer_correct: {pct(agree, len(shared))}, "
                  f"Cohen's kappa {cohen_kappa(a, b):.2f}")


# --------------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def instance_args(p):
        p.add_argument("--base-url", default="http://localhost:8000", help="MedRAG backend (default: %(default)s)")
        p.add_argument("--top-k", type=int, default=4)
        p.add_argument("--pause", type=float, default=2.0, help="seconds between LLM questions (free-tier limits)")
        p.add_argument("--keep", action="store_true", help="keep the account, uploaded documents and saved questions")
        p.add_argument("--email", default=os.environ.get("MEDRAG_EMAIL"),
                       help="sign in as this account instead of a throwaway one (or MEDRAG_EMAIL)")
        p.add_argument("--password", default=os.environ.get("MEDRAG_PASSWORD"), help="(or MEDRAG_PASSWORD)")

    smoke = sub.add_parser("smoke", help="hand-written smoke set (5 documents, 33 questions)")
    instance_args(smoke)

    pmq = sub.add_parser("pubmedqa", help="PubMedQA-L: expert-labelled questions over real PubMed abstracts")
    instance_args(pmq)
    pmq.add_argument("--n", type=int, default=1000, help="questions to use, up to 1000 (default: all)")
    pmq.add_argument("--holdout", type=float, default=0.1, help="fraction of abstracts left out (unanswerable)")
    pmq.add_argument("--answer-sample", type=int, default=60, help="questions to send to the LLM (0 = retrieval only)")
    pmq.add_argument("--seed", type=int, default=7)

    exp = sub.add_parser("review-export", help="write a CSV of answers for clinicians to grade")
    exp.add_argument("results", help="a results JSON written by smoke or pubmedqa")
    exp.add_argument("-o", "--output")

    summ = sub.add_parser("review-summary", help="summarise graded review CSVs (two sheets: inter-rater agreement)")
    summ.add_argument("sheets", nargs="+")

    args = parser.parse_args()
    {"smoke": run_smoke, "pubmedqa": run_pubmedqa, "review-export": review_export,
     "review-summary": review_summary}[args.command](args)


if __name__ == "__main__":
    main()
