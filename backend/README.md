# ScholarRAG Backend: Architecture & Design Decisions

This document provides a deep technical walkthrough of the core architectures, model selections, and logic implementation in the ScholarRAG pipeline.

---

## 🤖 Model Selection Analysis

We carefully selected specific models to optimize retrieval quality, comply with vector database limits, and ensure production reliability.

### 1. Embeddings: `BAAI/bge-large-en-v1.5` (1024-dim)
*   **Why we chose it**: 
    - The user's Pinecone Starter Plan is restricted to a single index. To maximize retrieval performance within a single index, we needed a state-of-the-art embedding model.
    - `bge-large-en-v1.5` consistently ranks at the top of the MTEB (Massive Text Embedding Benchmark) for retrieval tasks.
    - It outputs **1024 dimensions**, matching the index configuration, yielding a significantly higher representation capacity than smaller 384-dimensional models (like `all-MiniLM-L6-v2`).
*   **Why not others**: 
    - `all-MiniLM-L6-v2` is fast and lightweight but offers weaker retrieval recall on complex, dense academic text.
    - OpenAI's `text-embedding-3-small` or `text-embedding-3-large` are API-dependent, which introduces external latency and API cost. Using a local `SentenceTransformer` leverages the host system resources for free, offline embedding generation.

### 2. Large Language Model: `gemini-2.5-flash`
*   **Why we chose it**: 
    - `gemini-2.5-flash` is Google's latest lightweight, high-speed model that supports structured JSON response schemas directly.
    - It is the default supported model for modern API key formats (`AQ.Ab...`).
    - It provides a massive context window and has extremely low latency compared to legacy versions.
*   **Why not others**:
    - `gemini-1.5-flash` is deprecated or restricted under newer API keys, resulting in `404 model not found` errors.
    - `gpt-4o` or `claude-3-5-sonnet` require separate accounts, introduce higher token costs, and have lower request rate limits on their free tiers.

### 3. Reranker: Cohere Rerank API with local `CrossEncoder` fallback
*   **Why we chose it**:
    - **Dual-Layer Architecture**: Cohere's `rerank-english-v3.0` uses cross-attention to calculate relevance scores for document-query pairs, greatly improving retrieval precision.
    - **Local Fallback**: If `COHERE_API_KEY` is not provided or hits rate limits, the system falls back to a local `CrossEncoder` model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) running on the server.
*   **Why not others**:
    - Pure vector search only matches spatial similarity (bi-encoders), which often misses complex keyword overlaps. Reranking using cross-attention bridges this gap.

---

## ✂️ Semantic Chunking

Rather than dividing documents by an arbitrary number of tokens, which cuts off sentences mid-thought, ScholarRAG implements **Semantic Similarity Chunking**:

1.  **Sentence Tokenization**: The PDF is parsed, extracting pages and splitting raw text into complete, grammatically sound sentences.
2.  **Vector Distances**: We compute embeddings for all sentences and calculate the cosine distance between consecutive sentences.
3.  **Dynamic Thresholding**: The system determines distance thresholds dynamically using percentiles. Sentences are merged into a chunk until the similarity distance to the next sentence exceeds the calculated percentile threshold (indicating a topic transition).
4.  **Size Constraints**: We enforce hard limits (minimum of 2 sentences, maximum of 15 sentences per chunk) to ensure chunks are neither too small (losing context) nor too large (exceeding context limits).

---

## 🔍 Hybrid Search & RRF Fusion

We implement a multi-channel retrieval system combining dense vector search and sparse keyword search:

*   **Dense Search (Pinecone)**: Captures semantic meaning, synonyms, and conceptual intent.
*   **Sparse Search (BM25)**: Evaluates exact keyword matching (equations, acronyms, specific research names, terminology).
*   **Reciprocal Rank Fusion (RRF)**: Merges the candidate ranks from both vectors and BM25 to calculate a unified relevance score:
    $$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
    Where $M$ is the retrieval channels (Vector + BM25), $r_m(d)$ is the rank of document $d$ in channel $m$, and $k$ is a constant (default: 60) that dampens the influence of outlier high-rankings.

---

## 🏷️ Citation Enforcement & Stream Parsing

To prevent LLM hallucinations, ScholarRAG enforces strict citations:

1.  **Strict Schema**: The LLM is prompted with a structured JSON schema requiring an `answer` (using bracketed numbers like `[1]`, `[2]` for inline references) and a list of `citations` containing `source`, `page`, and `snippet`.
2.  **Incremental Parser**: The Gemini API streams JSON text. Our custom generator [`extract_streaming_answer`](file:///Users/kamallarishikeshthanay/Desktop/RAG-pipeline/backend/app/pipeline.py#L336) parses the raw incoming JSON chunks in real-time, extracting only the text inside the `"answer"` value and yielding it to the frontend SSE stream.
3.  **Post-Stream Compilation**: Once the text stream finishes, the server parses the final full JSON string to extract the `citations` list, streaming it as the final block to the frontend.

---

## 📈 Monitoring & Observability

We use **Langfuse** to trace every execution step of the RAG pipeline:

*   **Langfuse v4 Observation Model**: We wrapped all pipeline steps (Hybrid Retrieval, Reranking, Generation) with custom span observations using Langfuse's `start_as_current_observation` SDK models.
*   **Detailed Spans**: Every trace exposes metadata such as:
    - Raw documents fetched from Pinecone and BM25.
    - Rerank scores assigned by the cross-encoder.
    - Generation prompt variables, temperatures, and model versions.
    - Final parsed structured output and tokens count.

---

## 🧪 Quality Evaluation Framework (RAGAS)

To ensure the RAG application remains performant and accurate, we integrated **RAGAS** (Retrieval Augmented Generation Assessment):

*   **Supported Metrics**:
    - **Faithfulness**: Verifies if the generated answer is strictly supported by the retrieved context.
    - **Answer Relevancy**: Measures how directly the generated answer addresses the user's question.
    - **Context Recall**: Assesses if the retrieved context contains all ground truth facts.
    - **Context Precision**: Determines if the most relevant context chunks are ranked highest.
*   **Langfuse Integration**: The evaluation runs are instrumented with the Langfuse Langchain callback handler (`from langfuse.langchain import CallbackHandler`), automatically syncing evaluation metrics, inputs, and outputs to the Langfuse cloud dashboard for regression tracking.
