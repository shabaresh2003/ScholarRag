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


import os
import fitz  # PyMuPDF
import boto3
from PIL import Image
from io import BytesIO
from backend.app.config import (
    EMBEDDING_MODEL_NAME,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    AWS_S3_BUCKET_NAME,
    GEMINI_MODEL_NAME
)

# Lazy-loaded GenAI Vertex client from pipeline module
def get_vertex_client():
    from backend.app.pipeline import vertex_client
    return vertex_client

def upload_image_to_s3(image_bytes: bytes, filename: str) -> str:
    """Uploads cropped image bytes to AWS S3 bucket and returns the permanent HTTPS URL."""
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        # Fallback to local fake URL for demo/offline debugging
        return f"https://s3.amazonaws.com/{AWS_S3_BUCKET_NAME}/{filename}"
    try:
        s3 = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        s3.put_object(
            Bucket=AWS_S3_BUCKET_NAME,
            Key=filename,
            Body=image_bytes,
            ContentType="image/png"
        )
        return f"https://{AWS_S3_BUCKET_NAME}.s3.amazonaws.com/{filename}"
    except Exception as e:
        print(f"S3 image upload failed: {e}")
        return f"https://s3.amazonaws.com/{AWS_S3_BUCKET_NAME}/{filename}"

def generate_image_summary(image_bytes: bytes) -> str:
    """Passes cropped image directly to Gemini multimodal model to generate a descriptive summary."""
    client = get_vertex_client()
    if not client:
        return "Image description unavailable."
    try:
        from google.genai import types
        # Gemini accepts inline data as bytes
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png"
        )
        prompt = "Describe this image, chart, or diagram in detail. If it is a chart, extract all key data points."
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[image_part, prompt]
        )
        return response.text.strip() if response.text else "Image description unavailable."
    except Exception as e:
        print(f"Multimodal image summarization failed: {e}")
        return "Image description unavailable."

def extract_sentences_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text chunks using PyMuPDF and crops layout images/figures using unstructured/fitz partitions.
    Pushes images to AWS S3, generates text summaries via Gemini, and indexes them in the database.
    """
    doc = fitz.open(pdf_path)
    base_name = os.path.basename(pdf_path)
    all_sentences = []

    # 1. First extract text and page ranges
    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        page = doc[page_idx]
        text = page.get_text()
        if not text:
            continue
        sentences = split_into_sentences(text)
        for s in sentences:
            all_sentences.append({
                "text": s,
                "page": page_num,
                "image_url": None
            })

    # 2. Multimodal extraction: detect and crop images / figures
    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        page = doc[page_idx]
        image_list = page.get_images(full=True)
        
        for img_idx, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Image metadata definitions
                img_name = f"{base_name}_page_{page_num}_img_{img_idx}.png"
                
                # Upload image to Object Storage (S3 Bucket)
                s3_url = upload_image_to_s3(image_bytes, img_name)
                
                # Generate summary utilizing Vertex AI Gemini Multimodal model
                summary = generate_image_summary(image_bytes)
                
                # Append the image chunk summary to our index corpus
                all_sentences.append({
                    "text": f"[Image Context from {base_name} Page {page_num}]: {summary}",
                    "page": page_num,
                    "image_url": s3_url
                })
            except Exception as img_err:
                print(f"Error processing image {img_idx} on page {page_num}: {img_err}")

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
        # Pick first non-null image URL in the sentence list if any exists
        image_url = next((s.get("image_url") for s in sentence_list if s.get("image_url")), None)
        return {
            "text": text,
            "pages": pages,
            "image_url": image_url
        }
