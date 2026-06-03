import json
import os
import re
import uuid
from typing import List, Dict, Any, Generator, Optional, Tuple
import google.generativeai as genai
from pydantic import BaseModel

from backend.app.config import (
    GEMINI_API_KEY,
    COHERE_API_KEY,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST,
    GEMINI_MODEL_NAME,
    prompt_manager
)
from backend.app.database import get_db
from backend.app.schemas import CitationResponse

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ----------------------------------------------------
# TraceWrapper and SpanWrapper for Langfuse v4 compatibility
# ----------------------------------------------------
class SpanWrapper:
    def __init__(self, client, ctx_manager):
        self.client = client
        self.ctx_manager = ctx_manager
        self.span_obj = None

    def __enter__(self):
        self.span_obj = self.ctx_manager.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ctx_manager.__exit__(exc_type, exc_val, exc_tb)

    def span(self, name: str, input: Any = None):
        ctx = self.client.start_as_current_observation(
            as_type="span",
            name=name,
            input=input
        )
        return SpanWrapper(self.client, ctx)

    def generation(self, name: str, model: str = None, model_parameters: Dict = None, input: Any = None):
        ctx = self.client.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model,
            input=input
        )
        return SpanWrapper(self.client, ctx)

    def update(self, **kwargs):
        if self.span_obj:
            try:
                self.span_obj.update(**kwargs)
            except Exception:
                pass

class TraceWrapper:
    def __init__(self, client):
        self.client = client

    def trace(self, name: str, input: Any = None):
        ctx = self.client.start_as_current_observation(
            as_type="span",
            name=name,
            input=input
        )
        return SpanWrapper(self.client, ctx)

# ----------------------------------------------------
# Dummy classes for trace fallback when Langfuse is off
# ----------------------------------------------------
class DummySpan:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass
    def end(self, *args, **kwargs): pass
    def update(self, *args, **kwargs): pass

class DummyTrace:
    def span(self, *args, **kwargs): return DummySpan()
    def generation(self, *args, **kwargs): return DummySpan()
    def update(self, *args, **kwargs): pass

class DummyLangfuse:
    def trace(self, *args, **kwargs): return DummyTrace()

# ----------------------------------------------------
# Local Cross-Encoder Reranker
# ----------------------------------------------------
class LocalReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        if not docs:
            return []
        pairs = [(query, doc["text"]) for doc in docs]
        scores = self.model.predict(pairs)
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)
        # Sort descending by cross-encoder relevance
        sorted_docs = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
        return sorted_docs[:top_n]

# ----------------------------------------------------
# RAG Pipeline Implementation
# ----------------------------------------------------
class RAGPipeline:
    def __init__(self):
        # Initialize Langfuse
        if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
            from langfuse import Langfuse
            client = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST
            )
            self.langfuse = TraceWrapper(client)
        else:
            self.langfuse = DummyLangfuse()
            
        self.reranker = None

    def _get_reranker(self):
        if self.reranker is None:
            if COHERE_API_KEY:
                import cohere
                self.reranker = cohere.Client(api_key=COHERE_API_KEY)
            else:
                self.reranker = LocalReranker()
        return self.reranker

    def rerank_docs(self, query: str, docs: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        if not docs:
            return []
            
        reranker = self._get_reranker()
        if isinstance(reranker, LocalReranker):
            return reranker.rerank(query, docs, top_n)
        else:
            try:
                texts = [doc["text"] for doc in docs]
                response = reranker.rerank(
                    model="rerank-english-v3.0",
                    query=query,
                    documents=texts,
                    top_n=top_n
                )
                reranked = []
                for result in response.results:
                    idx = result.index
                    doc = docs[idx].copy()
                    doc["rerank_score"] = float(result.relevance_score)
                    reranked.append(doc)
                return reranked
            except Exception as e:
                print(f"Cohere Rerank failed, falling back to local: {e}")
                # Fallback to local cross-encoder
                self.reranker = LocalReranker()
                return self.reranker.rerank(query, docs, top_n)

    def run_query(self, query: str, source_filter: Optional[str] = None) -> Tuple[CitationResponse, List[Dict[str, Any]]]:
        """Runs vector+BM25 search, rerank, and Gemini generation. Fully traced."""
        trace = self.langfuse.trace(
            name="RAG-Query-Execution",
            input={"query": query, "source_filter": source_filter}
        )
        
        # 1. Retrieval
        with trace.span(name="Hybrid-Retrieval") as retrieval_span:
            retrieval_span.update(input={"query": query, "source_filter": source_filter})
            db_inst = get_db()
            raw_docs = db_inst.hybrid_search(query, top_k=20, source_filter=source_filter)
            retrieval_span.update(output={"raw_docs_count": len(raw_docs)})

        # 2. Reranking
        with trace.span(name="Reranking") as rerank_span:
            rerank_span.update(input={"docs_count": len(raw_docs)})
            reranked_docs = self.rerank_docs(query, raw_docs, top_n=5)
            rerank_span.update(output={"reranked_docs_count": len(reranked_docs)})

        # 3. Formulate Prompt
        active_version, system_prompt, user_template = prompt_manager.get_prompt()
        
        context_str = ""
        for idx, doc in enumerate(reranked_docs):
            context_str += f"Segment [{idx+1}] (Source: {doc['source']}, Pages: {doc['pages']}):\n{doc['text']}\n\n"
            
        user_prompt = user_template.format(query=query, context=context_str)
        
        # 4. Generate Answer using Gemini with schema enforcement
        llm_input = [
            {"role": "user", "parts": [user_prompt]}
        ]
        
        generation = trace.generation(
            name="Gemini-Citation-Generation",
            model=GEMINI_MODEL_NAME,
            model_parameters={"temperature": 0.0, "prompt_version": active_version},
            input=llm_input
        )
        
        try:
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL_NAME,
                system_instruction=system_prompt
            )
            
            response = model.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=CitationResponse,
                    temperature=0.0
                )
            )
            
            output_text = response.text
            generation.update(output=output_text)
            
            parsed_response = CitationResponse.model_validate_json(output_text)
            
            trace.update(output=parsed_response.model_dump())
            return parsed_response, reranked_docs
            
        except Exception as e:
            generation.update(output=f"Error: {str(e)}", metadata={"failed": True})
            trace.update(output=f"Error: {str(e)}")
            raise e
    def run_query_stream(self, query: str, source_filter: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """Runs the hybrid RAG query and streams the response incrementally (SSE ready)."""
        trace = self.langfuse.trace(
            name="RAG-Query-Streaming-Execution",
            input={"query": query, "source_filter": source_filter}
        )
        
        generation = None
        try:
            # 1. Retrieval
            with trace.span(name="Hybrid-Retrieval") as retrieval_span:
                db_inst = get_db()
                raw_docs = db_inst.hybrid_search(query, top_k=20, source_filter=source_filter)
                retrieval_span.update(output={"raw_docs_count": len(raw_docs)})

            # 2. Reranking
            with trace.span(name="Reranking") as rerank_span:
                reranked_docs = self.rerank_docs(query, raw_docs, top_n=5)
                rerank_span.update(output={"reranked_docs_count": len(reranked_docs)})

            # Stream documents list first to frontend so it can display references immediately
            yield {
                "type": "sources",
                "sources": reranked_docs
            }

            # 3. Formulate Prompt
            active_version, system_prompt, user_template = prompt_manager.get_prompt()
            
            context_str = ""
            for idx, doc in enumerate(reranked_docs):
                context_str += f"Segment [{idx+1}] (Source: {doc['source']}, Pages: {doc['pages']}):\n{doc['text']}\n\n"
                
            user_prompt = user_template.format(query=query, context=context_str)
            
            # 4. Generate streaming content
            generation = trace.generation(
                name="Gemini-Citation-Streaming",
                model=GEMINI_MODEL_NAME,
                model_parameters={"temperature": 0.0, "prompt_version": active_version},
                input=[{"role": "user", "parts": [user_prompt]}]
            )
            
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL_NAME,
                system_instruction=system_prompt
            )
            
            response_stream = model.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=CitationResponse,
                    temperature=0.0
                ),
                stream=True
            )
            
            # Read streaming JSON tokens and extract answer text incrementally
            full_json_str = ""
            for chunk in extract_streaming_answer(response_stream):
                yield {
                    "type": "text",
                    "text": chunk
                }
                
            # Once text is fully streamed, let's gather full response to parse citations
            # We fetch the full text from the generator
            full_json_str = response_stream.text
            generation.update(output=full_json_str)
            
            # Parse final JSON to extract formal citations
            try:
                parsed_json = json.loads(full_json_str)
                citations = parsed_json.get("citations", [])
                yield {
                    "type": "citations",
                    "citations": citations
                }
                trace.update(output=parsed_json)
            except Exception as parse_err:
                print(f"Error parsing final streaming JSON: {parse_err}")
                trace.update(output={"raw": full_json_str, "parse_error": str(parse_err)})
                
        except Exception as e:
            import traceback
            print("ERROR IN CHAT STREAM:")
            traceback.print_exc()
            if generation:
                try:
                    generation.update(output=f"Error: {str(e)}", metadata={"failed": True})
                except Exception:
                    pass
            trace.update(output=f"Error: {str(e)}")
            yield {
                "type": "error",
                "error": str(e)
            }
def extract_streaming_answer(response_stream) -> Generator[str, None, None]:
    """Helper generator to extract and stream only the 'answer' field from Gemini's structured JSON stream."""
    buffer = ""
    answer_started = False
    answer_ended = False
    yielded_idx = 0
    
    for chunk in response_stream:
        if not chunk.text:
            continue
        buffer += chunk.text
        
        if not answer_started:
            # Look for `"answer": "` key prefix
            idx = buffer.find('"answer":')
            if idx != -1:
                # Find the next quotation mark which starts the answer string
                quote_idx = buffer.find('"', idx + 9)
                if quote_idx != -1:
                    answer_started = True
                    yielded_idx = quote_idx + 1
                    
        if answer_started and not answer_ended:
            # Search for the closing quotation mark
            # (must not be escaped with a backslash)
            end_idx = -1
            i = yielded_idx
            while i < len(buffer):
                if buffer[i] == '"' and (i == 0 or buffer[i-1] != '\\'):
                    # Verify it's followed by a comma or closing bracket, or we're near the end
                    # (to avoid splitting early on nested quotes if any exist)
                    end_idx = i
                    break
                i += 1
                
            if end_idx != -1:
                # We reached the end of the answer field!
                yield buffer[yielded_idx:end_idx]
                answer_ended = True
            else:
                # Yield safely, leaving at least 2 chars in the buffer so we don't accidentally
                # output parts of the ending quote or an escape sequence
                safe_len = len(buffer) - yielded_idx
                if safe_len > 2:
                    yield_chunk = buffer[yielded_idx:len(buffer) - 2]
                    yielded_idx += len(yield_chunk)
                    yield yield_chunk
                    
    # Flush remaining answer text if we completed the stream but didn't locate the ending quote
    if answer_started and not answer_ended:
        yield buffer[yielded_idx:]
