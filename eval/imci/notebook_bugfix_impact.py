"""Measure the effect of the notebook bug fixes by running the original and fixed notebook code.

The original code is read from git history (commit 8635247, before the fixes). Hugging Face is not
needed: scispaCy biomedical word vectors stand in for the sentence-transformers model.

Usage (from the repo root): python eval/imci/notebook_bugfix_impact.py
"""
import io, json, re, subprocess, sys, tempfile, types, contextlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_COMMIT = "8635247"
S = tempfile.mkdtemp()
sys.path.insert(0, str(ROOT / "backend")); sys.path.insert(0, str(ROOT / "eval" / "imci"))
from run_retrieval_eval import normalise, spacy_encoder
from app.rag.loaders import extract_pages
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

encode = spacy_encoder("en_core_sci_md")
class SpacyST:  # stands in for SentenceTransformer (Hugging Face is unreachable here)
    def __init__(self, *a, **k): pass
    def get_sentence_embedding_dimension(self): return 200
    def encode(self, texts, **k): return np.asarray(encode(list(texts)), dtype=np.float32)

def mod(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m
Dummy = type("Dummy", (), {"__init__": lambda self, *a, **k: None})
mod("torch", cuda=types.SimpleNamespace(is_available=lambda: False))
mod("sentence_transformers", SentenceTransformer=SpacyST)
def no_pipeline(*a, **k): raise RuntimeError("no local LLM in this experiment")
mod("transformers", pipeline=no_pipeline)
mod("openai")
mod("IPython"); mod("IPython.display", display=lambda *a, **k: None, Markdown=str)
mod("langchain"); mod("langchain.text_splitter", RecursiveCharacterTextSplitter=RecursiveCharacterTextSplitter)
mod("langchain.schema", Document=Document); mod("langchain.prompts", PromptTemplate=PromptTemplate)
mod("langchain.chains", RetrievalQA=Dummy); mod("langchain_classic"); mod("langchain_classic.chains", RetrievalQA=Dummy)
mod("langchain_community"); mod("langchain_community.document_loaders", PyPDFLoader=Dummy, TextLoader=Dummy, Docx2txtLoader=Dummy)
mod("langchain_community.vectorstores", FAISS=Dummy, Chroma=Dummy)
mod("langchain_community.embeddings", HuggingFaceEmbeddings=Dummy)
mod("langchain_openai", OpenAI=Dummy, ChatOpenAI=Dummy)

def load(path, name):
    src = "\n".join("# " + l if l.startswith("!") else l for l in Path(path).read_text().splitlines())
    m = types.ModuleType(name); sys.modules[name] = m
    exec(compile(src, name, "exec"), m.__dict__)
    return m

def notebook_source(nb_json):
    return "".join(json.loads(nb_json)["cells"][0]["source"])

NOTEBOOK = "MedRag (1).ipynb"
Path(f"{S}/cell_old.py").write_text(notebook_source(
    subprocess.run(["git", "show", f"{ORIGINAL_COMMIT}:{NOTEBOOK}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout))
Path(f"{S}/cell_new.py").write_text(notebook_source((ROOT / NOTEBOOK).read_text(encoding="utf-8")))
old, new = load(f"{S}/cell_old.py", "nb_old"), load(f"{S}/cell_new.py", "nb_new")
questions = [json.loads(l) for l in (ROOT / "eval/imci/questions_imci.jsonl").read_text().splitlines()]
pages = [Document(page_content=pg.text, metadata={"page": pg.number, "source": "HBprelims.pdf"})
         for pg in extract_pages(ROOT / "HBprelims.pdf", ".pdf", ocr_enabled=False)]

def build(module, store, **cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        rag = module.MedicalDocumentRAG(config=module.RAGConfig(vector_store_type=store, **cfg), api_keys={})
        rag.documents = pages
        rag.process_and_chunk_document(); rag.create_embeddings_and_index()
    return rag

def run(rag, q):
    with contextlib.redirect_stdout(io.StringIO()):
        return rag.retrieve_relevant_chunks(q)

report = {}
demo = dict(chunk_size=512, chunk_overlap=128, top_k_retrieval=4)  # the notebook demo's settings

# 1. ChromaDB retrieval: old vs new
for label, module in (("old", old), ("new", new)):
    rag = build(module, "chromadb", **demo)
    ok = hits = 0
    for q in questions:
        res = run(rag, q["question"])
        ok += bool(res)
        hits += any(normalise(q["answer_phrase"]) in normalise(c.page_content) for c, _, _ in res)
    report[f"chroma_{label}"] = {"queries_returning_results": ok, "recall@4": hits / len(questions)}

# 2. FAISS with top_k larger than the number of chunks (original RAGConfig defaults: 1000/200, top_k=5)
sample = Document(page_content=Path(new.create_sample_document(f"{S}/sample.txt")).read_text(), metadata={})
for label, module in (("old", old), ("new", new)):
    with contextlib.redirect_stdout(io.StringIO()):
        rag = module.MedicalDocumentRAG(config=module.RAGConfig(), api_keys={})
        rag.documents = [sample]; rag.process_and_chunk_document(); rag.create_embeddings_and_index()
        res = rag.retrieve_relevant_chunks("What are the diagnostic criteria?")
    ids = [id(c) for c, _, _ in res]
    report[f"faiss_small_doc_{label}"] = {"chunks_in_index": len(rag.text_chunks), "results": len(res),
                                          "duplicate_results": len(ids) - len(set(ids))}

# 3. No-LLM fallback excerpt: same FAISS retrieval, old vs new line selection
rag = build(new, "faiss", **demo)
shown = {"old": 0, "new": 0}; lines_kept = {"old": [], "new": []}; answerable = 0
for q in questions:
    res = run(rag, q["question"])
    context = "\n\n".join(f"[Source {i+1}]: {c.page_content}" for i, (c, _, _) in enumerate(res))
    phrase = normalise(q["answer_phrase"])
    if phrase not in normalise(context):
        continue
    answerable += 1
    total = sum(1 for l in context.split("\n") if l.strip())
    for label, module in (("old", old), ("new", new)):
        summary = module.MedicalDocumentRAG._create_summary_from_context(rag, context, q["question"])
        shown[label] += phrase in normalise(summary.replace("*", ""))
        matched = sum(1 for l in context.split("\n") if l.strip() and (
            (lambda qw, lw: bool(qw & lw) or any(w in l.lower() for w in qw))(set(q["question"].lower().split()), set(l.lower().split()))
            if label == "old" else
            bool({w for w in re.findall(r"[a-z0-9]+", q["question"].lower()) if len(w) > 2} & set(re.findall(r"[a-z0-9]+", l.lower())))))
        lines_kept[label].append(matched / total)
report["fallback"] = {"questions_with_answer_in_context": answerable,
                      "answer_shown_old": shown["old"], "answer_shown_new": shown["new"],
                      "avg_share_of_lines_matched_old": round(float(np.mean(lines_kept["old"])), 3),
                      "avg_share_of_lines_matched_new": round(float(np.mean(lines_kept["new"])), 3)}
print(json.dumps(report, indent=2))
(ROOT / "eval" / "imci" / "notebook_bugfix_results.json").write_text(json.dumps(report, indent=2))
