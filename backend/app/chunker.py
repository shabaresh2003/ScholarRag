import re
from typing import List, Dict, Any
import numpy as np
from pypdf import PdfReader
from backend.app.config import EMBEDDING_MODEL_NAME

def split_into_sentences(text: str) -> List[str]:
    """Splits a body of text into sentences using a robust regex pattern."""
    # Clean whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    # Sentence splitter regex: splits on punctuation (.!?), ignoring common academic abbreviations
    # Separated by length to comply with Python's fixed-width look-behind requirement
    abbr_len2 = r'\b(?:Dr|Mr|Ms|vs|eg|ie|al)'
    abbr_len3 = r'\b(?:Mrs|etc|Fig|Vol|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
    abbr_len4 = r'\b(?:Prof|Figs)'
    
    sentence_end = re.compile(
        rf'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<!{abbr_len2}\.)(?<!{abbr_len3}\.)(?<!{abbr_len4}\.)(?<=\.|\?|\!)\s'
    )
    sentences = sentence_end.split(text)
    return [s.strip() for s in sentences if len(s.strip()) > 3]


def extract_sentences_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extracts text from PDF page by page and tokenizes into sentences with page numbers."""
    reader = PdfReader(pdf_path)
    all_sentences = []
    
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        text = page.extract_text()
        if not text:
            continue
        sentences = split_into_sentences(text)
        for s in sentences:
            all_sentences.append({
                "text": s,
                "page": page_num
            })
    return all_sentences


class SemanticChunker:
    def __init__(
        self, 
        model_name: str = EMBEDDING_MODEL_NAME, 
        similarity_threshold_percentile: float = 25.0,
        min_sentences_per_chunk: int = 2,
        max_sentences_per_chunk: int = 8,
        min_similarity_split: float = 0.45,
        max_similarity_merge: float = 0.85
    ):
        self.model_name = model_name
        self.similarity_threshold_percentile = similarity_threshold_percentile
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.min_similarity_split = min_similarity_split
        self.max_similarity_merge = max_similarity_merge
        self._model = None

    @property
    def model(self):
        if self._model is None:
            # Lazy load model to speed up server startups
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def chunk_sentences(self, sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Splits sentences into chunks based on embedding similarity & constraints."""
        if not sentences:
            return []
        
        if len(sentences) == 1:
            return [self._create_chunk_dict(sentences)]
            
        texts = [s["text"] for s in sentences]
        # Embed all sentences
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Calculate cosine similarity between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            emb1 = embeddings[i]
            emb2 = embeddings[i+1]
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            similarities.append(similarity)
            
        # Determine the similarity threshold from percentile
        percentile_threshold = np.percentile(similarities, self.similarity_threshold_percentile)
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i, similarity in enumerate(similarities):
            next_sentence = sentences[i+1]
            
            # Check constraints to decide if we should split
            should_split = False
            
            # 1. Similarity is below the dynamic threshold
            if similarity < percentile_threshold:
                should_split = True
            
            # 2. Hard threshold bounds
            if similarity < self.min_similarity_split:
                should_split = True
            elif similarity > self.max_similarity_merge:
                should_split = False
                
            # 3. Size constraints override similarity logic
            if len(current_chunk) >= self.max_sentences_per_chunk:
                should_split = True
            elif len(current_chunk) < self.min_sentences_per_chunk:
                should_split = False
                
            if should_split:
                chunks.append(self._create_chunk_dict(current_chunk))
                current_chunk = [next_sentence]
            else:
                current_chunk.append(next_sentence)
                
        # Append final chunk
        if current_chunk:
            chunks.append(self._create_chunk_dict(current_chunk))
            
        return chunks

    def _create_chunk_dict(self, sentence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        text = " ".join([s["text"] for s in sentence_list])
        pages = list(sorted(list(set([s["page"] for s in sentence_list]))))
        return {
            "text": text,
            "pages": pages
        }
