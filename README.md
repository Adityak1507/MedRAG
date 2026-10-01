

## MedRag: Medical Document Retrieval-Augmented Generation

**MedRag** is a powerful Retrieval-Augmented Generation (RAG) system specifically designed for querying and analyzing medical documents. It allows users to upload custom PDF, DOCX, or TXT files and ask questions, receiving answers grounded **only** in the provided document's content.

This project is built using the **LangChain** framework and leverages modern embedding models and vector databases to ensure high-accuracy, context-specific responses from large medical texts.

-----

## Features

  * **Context-Specific QA:** Acts as a medical assistant, answering questions strictly based on the provided document context to prevent hallucinations or use of external knowledge.
  * **Multi-Format Document Support:** Easily ingest PDF, TXT, and DOCX files (legacy `.doc` is not supported).
  * **Flexible Vector Store Options:** Supports integration with multiple vector databases:
      * **FAISS** (default for local, fast indexing)
      * **ChromaDB**
      * **Pinecone** (requires API Key)
  * **Multiple LLM Backends:** Use your preferred Large Language Model (LLM) for generation:
      * **OpenAI**
      * **Anthropic**
      * **Cohere**
      * **Local** HuggingFace models (e.g., `distilgpt2`)
  * **Retrieval Customization:** Configurable document chunking (`chunk_size`, `chunk_overlap`) and retrieval size (`top_k_retrieval`).
  * **LangChain QA Chain Option:** Includes an alternative mode to switch to a standard `RetrievalQA` chain implementation.

-----

## Web App (FastAPI + React + PostgreSQL)

The same RAG pipeline is also available as a web application:

| Part | Stack | Location |
| :--- | :--- | :--- |
| **Backend** | FastAPI, SQLAlchemy, LangChain (splitting + chat models), sentence-transformers | `backend/` |
| **Database / vector store** | PostgreSQL 16 with **pgvector** (HNSW index, cosine distance) | `docker-compose.yml` |
| **Frontend** | React + TypeScript (Vite), served by nginx in Docker | `frontend/` |

Documents, chunks with their embeddings, and question history are all stored in Postgres, so they persist across restarts and you can search several documents at once.

### Run with Docker

```bash
cp .env.example .env        # optionally add OPENAI_API_KEY / ANTHROPIC_API_KEY / COHERE_API_KEY
docker compose up --build
```

- App: http://localhost:8080
- API docs (Swagger): http://localhost:8000/docs

The first start downloads the `all-MiniLM-L6-v2` embedding model from Hugging Face, and it is cached in a Docker volume after that. With no LLM key set, answers are retrieval-only excerpts.

### Run locally for development

```bash
# Database: Postgres 16 with pgvector on localhost:5432
docker compose up db -d

# Backend
cd backend
pip install -r requirements-dev.txt
export DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/medrag
uvicorn app.main:app --reload            # http://localhost:8000

# Frontend (proxies /api to the backend)
cd frontend
npm install
npm run dev                              # http://localhost:5173
```

### API

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `POST` | `/api/documents` | Upload a PDF, DOCX or TXT file (multipart field `file`). Processing runs in the background and the status goes `processing` → `ready` / `failed`. |
| `GET` | `/api/documents` | List documents with their status and chunk counts. |
| `GET` / `DELETE` | `/api/documents/{id}` | Get a document, or delete it along with its chunks. |
| `POST` | `/api/query` | `{"question": "...", "document_ids": [1, 2], "top_k": 4}`. Returns the answer, the LLM used, and the sources. `document_ids` and `top_k` are optional. |
| `GET` | `/api/queries` | Recent questions and answers. |
| `GET` | `/api/info`, `/api/health` | Active configuration and counts, and a liveness check. |

### Configuration

The backend reads environment variables (or `backend/.env`):

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/medrag` | Postgres connection string. |
| `LLM_PROVIDER` | `auto` | `auto` (OpenAI → Anthropic → Cohere, whichever has a key), a specific provider, or `none`. |
| `OPENAI_MODEL` / `ANTHROPIC_MODEL` / `COHERE_MODEL` | `gpt-4o-mini` / `claude-opus-5-5` / `command-r` | Chat model for each provider. |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `all-MiniLM-L6-v2` / `384` | Must match each other. Changing them requires re-uploading documents. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `TOP_K` | `512` / `128` / `4` | Chunking and retrieval. |
| `MAX_UPLOAD_MB` | `25` | Upload size limit. |

### Tests

```bash
cd backend
pytest        # needs Postgres with pgvector; set DATABASE_URL to a disposable database (default: medrag_test)
```

The tests use a stand-in embedder and LLM, so no model download or API key is needed. **They drop and recreate all tables** in the target database.

-----

## Notebook: Installation and Setup

### 1\. Dependencies

This project requires Python and uses several key libraries. The dependencies can be installed using `pip`.

```bash
pip install langchain langchain-classic langchain-openai langchain-community
pip install faiss-cpu sentence-transformers chromadb pypdf docx2txt transformers torch
# Optional dependencies if using other models/stores
# pip install pinecone cohere anthropic langchain-anthropic
```

The necessary core libraries found in the project include:

  * `langchain`
  * `langchain-openai`
  * `langchain-community`
  * `faiss-cpu`
  * `sentence-transformers`
  * `chromadb`

### 2\. API Keys (If needed)

The system is designed to use external LLMs and vector stores, which require API keys. Set up your keys as environment variables or pass them directly as a dictionary to the `MedicalDocumentRAG` class initialization.

| Service | Key Variable Name |
| :--- | :--- |
| **OpenAI** | `OPENAI_API_KEY` |
| **Anthropic** | `ANTHROPIC_API_KEY` |
| **Cohere** | `COHERE_API_KEY` |
| **Pinecone** | `PINECONE_API_KEY` |

The notebook reads keys from environment variables and only prompts for the ones that are not set.

-----

## Usage

The project is structured as a **Jupyter Notebook** (`MedRag (1).ipynb`) and can be run step-by-step.

# Example Configuration from the notebook
config = RAGConfig(
    chunk_size=512,
    chunk_overlap=128,
    top_k_retrieval=4,
    temperature=0.1,
    vector_store_type="faiss" # Options: faiss, chromadb, pinecone
)
rag_system = MedicalDocumentRAG(config=config, api_keys=api_keys)
### Step 1: Configuration

Initialize the system by defining a configuration, including the vector store type (default is `faiss`) and retrieval settings.

### Step 2: Document Loading

You can choose to upload your own document or use a sample text for testing:

1.  **Upload:** Enter the path to your document (`.pdf`, `.docx`, or `.txt`) when prompted.
2.  **Sample Text:** The notebook includes a built-in sample text on **Chronic Fatigue Syndrome (CFS)** for quick testing.

The system automatically performs document splitting and chunking using the configured parameters.

### Step 3: Create Embeddings and Index

After loading the document, create the vector index. The system uses a **HuggingFace Embedding Model** (`all-MiniLM-L6-v2` by default) for creating embeddings.

# Runs automatically in the main loop if 'p' (Process) is selected
rag_system.create_embeddings_and_index()

### Step 4: Ask Questions (QA Loop)

Enter the main loop to ask medical questions about the document.

The output will provide the **Answer**, the **LLM Type** used, the **Number of Retrieved Chunks**, and the **Source Documents** used to formulate the answer, including page and source file information.

```
> Enter your question: What are the three core diagnostic symptoms for Chronic Fatigue Syndrome?
```

Type `stats` to see system statistics, `chain` to toggle the LangChain QA chain, or `q` to quit.

**Note:** If no LLM is initialized, the system defaults to a **retrieval-only** mode and will return the most relevant document excerpts without generating a synthetic answer.

-----

## RAG Configuration Details

The `RAGConfig` class allows you to fine-tune the system's performance:

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `embedding_model` | `all-MiniLM-L6-v2` | The HuggingFace model for creating embeddings. |
| `vector_store_type` | `"faiss"` | Vector store backend: `faiss`, `chromadb`, or `pinecone`. |
| `chunk_size` | `1000` | The maximum size of text chunks for indexing (the demo uses `512`). |
| `chunk_overlap` | `200` | The overlap between consecutive text chunks (the demo uses `128`). |
| `top_k_retrieval` | `5` | The number of most relevant documents to retrieve for the LLM (the demo uses `4`). |
| `temperature` | `0.1` | The LLM generation temperature (not sent to Anthropic models, which don't accept it). |
| `max_tokens` | `500` | Max response length for LLM generation. |

The LLM is chosen automatically from the API keys provided, in the order OpenAI → Anthropic → Cohere → local `distilgpt2`. The Pinecone index is named `medical-rag-index`.

-----

##  Contributing

Contributions are welcome\! If you have suggestions for new LLMs, vector stores, or document loaders, please submit a pull request.
