from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. Initialize the Open-Source Embedding Model (runs locally/on your compute)
# BAAI/bge models are highly optimized for RAG and retrieval tasks.
dense_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# 2. Configure the Semantic Chunker
text_splitter = SemanticChunker(
    dense_embeddings, 
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=85
)

# Raw data extracted from PDF parsing
raw_pdf_pages = [
    {"text": "Section 4.1: SIP Step-Up mandates a 10% annual increase for high-risk funds...", "page": 4, "category": "Compliance"},
    {"text": "Mutual Fund Risk Disclosure: Past performance does not guarantee future results...", "page": 12, "category": "Factsheet"}
]

# Generate semantic chunks and attach hard metadata
processed_chunks = []
for page in raw_pdf_pages:
    chunks = text_splitter.split_text(page["text"])
    for i, chunk_text in enumerate(chunks):
        doc = Document(
            page_content=chunk_text,
            metadata={
                "source": "financial_guidelines_2026.pdf",
                "page_number": page["page"],
                "category": page["category"]
            }
        )
        processed_chunks.append(doc)