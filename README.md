# ScholarRAG 🎓

ScholarRAG is a production-level, citation-enforced RAG (Retrieval-Augmented Generation) application designed for students and researchers to study research papers. The system reads uploaded PDFs, chunks them semantically, performs hybrid retrieval, reranks context, and generates precise, citation-backed answers while providing full observability tracing and quality evaluations.

---

## 🚀 Key Features

*   **Semantic Chunking**: Groups sentences based on embedding distance percentiles rather than arbitrary token splits, preserving context.
*   **Hybrid Retrieval (BM25 + Vector)**: Combines dense vector search (Pinecone Cloud) with keyword search (local BM25) fused using Reciprocal Rank Fusion (RRF).
*   **Context Reranking**: Re-evaluates context relevance using the Cohere Rerank API (with a local `CrossEncoder` fallback).
*   **Strict Citation Enforcement**: Gemini outputs a validated JSON schema containing citations (source filename, page numbers, exact quote snippets).
*   **Streaming SSE Interface**: Progressively streams the generated text in real-time, displaying citations and sources dynamically in the UI.
*   **Multimodal Ingestion & Retrieval**: Automatically extracts inline diagrams/images from PDFs, uploads them to AWS S3 (creating buckets automatically if missing), runs detailed multimodal summarization via Gemini 2.5 Flash, and retrieves/renders visual diagrams inline in the chat interface.
*   **Observability & Tracing**: Instruments every phase (retrieval, reranking, generation) to Langfuse.
*   **RAGAS Evaluation Framework**: Contains a quality evaluation suite that syncs performance scores to Langfuse.

---

## 📂 Project Structure

```text
RAG-pipeline/
├── backend/
│   ├── app/
│   │   ├── chunker.py       # Sentence splitter and semantic chunker
│   │   ├── config.py        # Settings, directory configs, prompt loading
│   │   ├── database.py      # Pinecone client, BM25 indexing, RRF fusion
│   │   ├── main.py          # FastAPI endpoints (upload, delete, chat stream)
│   │   ├── pipeline.py      # Core RAG execution, SSE stream parser, tracing
│   │   └── schemas.py       # Pydantic schemas (CitationResponse, QueryRequest)
│   ├── config/
│   │   └── prompts.yaml     # Hot-reloadable prompt versions config
│   ├── evaluate.py          # RAGAS evaluation runner
│   ├── run.py               # Backend startup script
│   └── tests/               # Pytest suite
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # React app with citation badges and SSE client
│   │   └── index.css        # Premium Vanilla CSS styling
│   └── package.json
└── .gitignore
```

---

## 🛠️ Quick Start

### 1. Configure Credentials
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY="your-gemini-key"
PINECONE_API_KEY="your-pinecone-key"
PINECONE_INDEX_NAME="research-papers"

# Optional (falls back to local CrossEncoder if empty)
COHERE_API_KEY="your-cohere-key"

# AWS Bedrock & S3 Configuration (for Bedrock routing & multimodal image pipeline)
AWS_ACCESS_KEY_ID="your-aws-access-key"
AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
AWS_REGION="us-east-1"
AWS_S3_BUCKET_NAME="scholar-rag-images"

# Optional (for monitoring & evaluations)
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_BASE_URL="https://us.cloud.langfuse.com"
```

### 2. Start the Backend
```bash
# Activate virtual environment
source .venv/bin/activate

# Navigate to backend and run
cd backend
python3 run.py
```
*Backend server will start at `http://localhost:8000`.*

### 3. Start the Frontend
In a new terminal:
```bash
cd frontend
npm run dev
```
*Frontend interface will start at `http://localhost:5173`.*

### 4. Run Evaluations
Run the RAGAS metrics suite:
```bash
PYTHONPATH=. .venv/bin/python backend/evaluate.py
```

---

For technical details regarding model choices, semantic chunking, and observability configurations, refer to the [**Backend Technical Architecture Documentation**](file:///Users/kamallarishikeshthanay/Desktop/RAG-pipeline/backend/README.md).
