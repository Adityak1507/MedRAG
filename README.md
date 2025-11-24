

## MedRag: Medical Document Retrieval-Augmented Generation

**MedRag** is a powerful Retrieval-Augmented Generation (RAG) system specifically designed for querying and analyzing medical documents. It allows users to upload custom PDF, DOCX, or TXT files and ask questions, receiving answers grounded **only** in the provided document's content.

This project is built using the **LangChain** framework and leverages modern embedding models and vector databases to ensure high-accuracy, context-specific responses from large medical texts.

-----

## Features

  * **Context-Specific QA:** Acts as a medical assistant, answering questions strictly based on the provided document context to prevent hallucinations or use of external knowledge.
  * **Multi-Format Document Support:** Easily ingest PDF, TXT, and DOCX files.
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

## Installation and Setup

### 1\. Dependencies

This project requires Python and uses several key libraries. The dependencies can be installed using `pip`.

```bash
pip install langchain langchain-openai langchain-community
pip install faiss-cpu sentence-transformers chromadb
# Optional dependencies if using other models/stores
# pip install pinecone-client cohere anthropic
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

The notebook attempts to load keys from a `.env` file or environment variables.

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
Enter your query (or 'h' for help, 'q' to quit): What are the three core diagnostic symptoms for Chronic Fatigue Syndrome?
```

**Note:** If no LLM is initialized, the system defaults to a **retrieval-only** mode and will return the most relevant document excerpts without generating a synthetic answer.

-----

## RAG Configuration Details

The `RAGConfig` class allows you to fine-tune the system's performance:

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `llm_type` | `"local"` | LLM provider: `openai`, `anthropic`, `cohere`, or `local`. |
| `llm_model` | `distilgpt2` (local) | The specific model name to use for generation. |
| `embedding_model` | `all-MiniLM-L6-v2` | The HuggingFace model for creating embeddings. |
| `vector_store_type` | `"faiss"` | Vector store backend: `faiss`, `chromadb`, or `pinecone`. |
| `chunk_size` | `512` | The maximum size of text chunks for indexing. |
| `chunk_overlap` | `128` | The overlap between consecutive text chunks. |
| `top_k_retrieval` | `4` | The number of most relevant documents to retrieve for the LLM. |
| `temperature` | `0.1` | The LLM generation temperature. |
| `max_tokens` | `300` | Max response length for LLM generation. |
| `index_name` | `medrag-index` | Name for remote vector store index (e.g., Pinecone). |

-----

##  Contributing

Contributions are welcome\! If you have suggestions for new LLMs, vector stores, or document loaders, please submit a pull request.
