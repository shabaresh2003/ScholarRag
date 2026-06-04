import os
import json
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from backend.app.config import DATA_DIR
from backend.app.database import get_db
from backend.app.chunker import SemanticChunker, extract_sentences_from_pdf
from backend.app.pipeline import RAGPipeline
from backend.app.schemas import QueryRequest, CitationResponse

# Initialize FastAPI App
app = FastAPI(
    title="Academic Research RAG API",
    description="A production-level RAG backend for research paper analysis with citations and tracing.",
    version="1.0.0"
)

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend domain (e.g. http://localhost:5173)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload directory exists
UPLOAD_DIR = DATA_DIR / "documents"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------
# Endpoints
# ----------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "Academic Research RAG API is running!"}

@app.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF document, chunks it semantically, and indexes it in the DB."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    # Sanitize filename
    safe_filename = "".join([c for c in file.filename if c.isalnum() or c in ('.', '_', '-')])
    file_path = UPLOAD_DIR / safe_filename
    
    try:
        # Save file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Extract sentences from PDF
        sentences = extract_sentences_from_pdf(str(file_path))
        if not sentences:
            raise HTTPException(status_code=400, detail="No readable text found in the PDF.")
            
        # 2. Perform Semantic Chunking
        chunker = SemanticChunker()
        chunks = chunker.chunk_sentences(sentences)
        
        # 3. Add to DB
        database = get_db()
        database.add_document(safe_filename, chunks)
        
        return {
            "filename": safe_filename,
            "chunks_count": len(chunks),
            "status": "successfully_indexed"
        }
    except Exception as e:
        # Cleanup file if indexing failed
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error indexing PDF: {str(e)}")

@app.get("/documents")
def list_documents():
    """Lists all research papers loaded into the system."""
    try:
        database = get_db()
        docs = database.get_all_document_names()
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Deletes a research paper and its associated chunks from vector and BM25 database."""
    try:
        database = get_db()
        all_docs = database.get_all_document_names()
        if filename not in all_docs:
            raise HTTPException(status_code=404, detail="Document not found.")
            
        # Delete from Chroma & rebuild BM25
        database.delete_document(filename)
        
        # Delete PDF file
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            os.remove(file_path)
            
        return {"message": f"Successfully deleted {filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    """Sync chat endpoint that returns structured citations and retrieved sources directly."""
    try:
        pipeline = RAGPipeline()
        answer, sources = pipeline.run_query(request.query, request.source_filter)
        return {
            "answer": answer.answer,
            "citations": [c.model_dump() for c in answer.citations],
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
def chat_stream_endpoint(request: QueryRequest):
    """Real-time streaming chat endpoint yielding text, sources, and citations via SSE."""
    try:
        pipeline = RAGPipeline()
        
        def event_generator():
            generator = pipeline.run_query_stream(request.query, request.source_filter)
            for chunk in generator:
                yield {
                    "event": chunk["type"],
                    "data": json.dumps(chunk)
                }
                
        return EventSourceResponse(event_generator())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
