import json
import os
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from pinecone import Pinecone, ServerlessSpec

from backend.app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL_NAME,
    METADATA_STORE_PATH
)

# Helper functions for local metadata store (BM25 and document registry)
def load_metadata_store() -> Dict[str, List[Dict[str, Any]]]:
    if not METADATA_STORE_PATH.exists():
        return {}
    with open(METADATA_STORE_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def save_metadata_store(data: Dict[str, List[Dict[str, Any]]]):
    with open(METADATA_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

class PineconeDatabase:
    def __init__(self):
        # 1. Initialize local SentenceTransformer model
        from sentence_transformers import SentenceTransformer
        self.emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # 2. Initialize Pinecone client
        self.api_key = PINECONE_API_KEY
        if not self.api_key:
            print("WARNING: PINECONE_API_KEY is not set! Pinecone operations will fail.")
            self.pc = None
            self.index = None
        else:
            self.pc = Pinecone(api_key=self.api_key)
            self._init_index()
            
        # 3. BM25-related fields
        self.bm25 = None
        self.bm25_docs = []  # Flat list of all chunks for BM25 search
        self.build_bm25_index()

    def _init_index(self):
        """Initializes the index, creating it if it doesn't already exist."""
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            if PINECONE_INDEX_NAME not in existing_indexes:
                print(f"Index '{PINECONE_INDEX_NAME}' not found in Pinecone. Creating it...")
                self.pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=1024,  # BAAI/bge-large-en-v1.5 is 1024 dimensions
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            self.index = self.pc.Index(PINECONE_INDEX_NAME)
        except Exception as e:
            print(f"Error initializing Pinecone index: {e}")
            self.index = None

    def build_bm25_index(self):
        """Rebuild BM25 index using chunks loaded from local metadata store."""
        try:
            store = load_metadata_store()
            self.bm25_docs = []
            tokenized_corpus = []
            
            for filename, chunks in store.items():
                for idx, chunk in enumerate(chunks):
                    doc_id = f"{filename}_{idx}"
                    doc_dict = {
                        "id": doc_id,
                        "text": chunk["text"],
                        "source": filename,
                        "pages": chunk["pages"]
                    }
                    self.bm25_docs.append(doc_dict)
                    tokenized_corpus.append(tokenize_text(chunk["text"]))
                    
            if tokenized_corpus:
                self.bm25 = BM25Okapi(tokenized_corpus)
            else:
                self.bm25 = None
        except Exception as e:
            print(f"Error building BM25 index: {e}")
            self.bm25 = None
            self.bm25_docs = []

    def add_document(self, filename: str, chunks: List[Dict[str, Any]]):
        """Generates embeddings, upserts vectors to Pinecone, and caches metadata locally."""
        if not chunks:
            return
            
        # 1. Update local metadata store
        store = load_metadata_store()
        store[filename] = chunks
        save_metadata_store(store)
        
        # 2. Upsert to Pinecone
        if self.index:
            try:
                vectors = []
                for idx, chunk in enumerate(chunks):
                    chunk_id = f"{filename}_{idx}"
                    emb = self.emb_model.encode(chunk["text"]).tolist()
                    vectors.append({
                        "id": chunk_id,
                        "values": emb,
                        "metadata": {
                            "source": filename,
                            "text": chunk["text"],
                            "pages": json.dumps(chunk["pages"]),
                            "chunk_idx": idx
                        }
                    })
                
                # Batch upsert in sizes of 100
                for i in range(0, len(vectors), 100):
                    self.index.upsert(vectors=vectors[i:i+100])
            except Exception as e:
                print(f"Error upserting vectors to Pinecone: {e}")
                
        # 3. Rebuild BM25 index
        self.build_bm25_index()

    def get_all_document_names(self) -> List[str]:
        """Returns list of unique document filenames from the registry."""
        store = load_metadata_store()
        return sorted(list(store.keys()))

    def delete_document(self, filename: str):
        """Deletes a document from Pinecone index and local registry."""
        # 1. Delete from Pinecone
        if self.index:
            try:
                self.index.delete(filter={"source": {"$eq": filename}})
            except Exception as e:
                print(f"Error deleting vectors from Pinecone: {e}")
                
        # 2. Delete from local metadata store
        store = load_metadata_store()
        if filename in store:
            del store[filename]
            save_metadata_store(store)
            
        # 3. Rebuild BM25 index
        self.build_bm25_index()

    def hybrid_search(self, query: str, top_k: int = 10, source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform hybrid search: Pinecone Vector search + BM25, combined using RRF."""
        # 1. Vector Search
        vector_results = []
        if self.index:
            try:
                query_embedding = self.emb_model.encode(query).tolist()
                filter_dict = {}
                if source_filter:
                    filter_dict = {"source": {"$eq": source_filter}}
                    
                vector_raw = self.index.query(
                    vector=query_embedding,
                    top_k=min(top_k * 3, 50),
                    filter=filter_dict if filter_dict else None,
                    include_metadata=True
                )
                
                for match in vector_raw.get("matches", []):
                    meta = match.get("metadata", {})
                    pages = []
                    if "pages" in meta:
                        try:
                            pages = json.loads(meta["pages"])
                        except Exception:
                            if isinstance(meta["pages"], str):
                                pages = [int(p) for p in meta["pages"].split(",") if p.strip()]
                    
                    vector_results.append({
                        "id": match["id"],
                        "text": meta.get("text", ""),
                        "source": meta.get("source", "unknown"),
                        "pages": pages,
                        "vector_score": float(match["score"])
                    })
            except Exception as e:
                print(f"Pinecone query failed: {e}")

        # 2. BM25 Search
        bm25_results = []
        if self.bm25:
            # Filter BM25 corpus matching source_filter
            filtered_docs_indices = []
            filtered_docs = []
            for i, doc in enumerate(self.bm25_docs):
                if not source_filter or doc["source"] == source_filter:
                    filtered_docs_indices.append(i)
                    filtered_docs.append(doc)
            
            if filtered_docs:
                tokenized_query = tokenize_text(query)
                scores = self.bm25.get_scores(tokenized_query)
                
                filtered_scores = [(scores[idx], filtered_docs[i]) for i, idx in enumerate(filtered_docs_indices)]
                filtered_scores.sort(key=lambda x: x[0], reverse=True)
                
                for score, doc in filtered_scores[:min(top_k * 3, 50)]:
                    if score > 0.0:
                        doc_copy = doc.copy()
                        doc_copy["bm25_score"] = float(score)
                        bm25_results.append(doc_copy)

        # 3. Reciprocal Rank Fusion (RRF)
        fused = reciprocal_rank_fusion(vector_results, bm25_results)
        
        return fused[:top_k]

def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]], 
    bm25_results: List[Dict[str, Any]], 
    k: int = 60
) -> List[Dict[str, Any]]:
    rrf_scores = {}
    doc_map = {}
    
    for rank, doc in enumerate(vector_results):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
    for rank, doc in enumerate(bm25_results):
        doc_id = doc["id"]
        if doc_id in doc_map:
            doc_map[doc_id]["bm25_score"] = doc.get("bm25_score")
        else:
            doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id]
        doc["rrf_score"] = rrf_scores[doc_id]
        results.append(doc)
        
    return results

def tokenize_text(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())

# Singleton database instance
db = None
def get_db():
    global db
    if db is None:
        db = PineconeDatabase()
    return db
