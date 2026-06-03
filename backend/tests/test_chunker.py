import pytest
from backend.app.chunker import split_into_sentences, SemanticChunker

def test_split_into_sentences():
    text = "The Transformer is a deep learning model. It was introduced in 2017 by Google. What is it used for? It is used for NLP!"
    sentences = split_into_sentences(text)
    
    assert len(sentences) == 4
    assert sentences[0] == "The Transformer is a deep learning model."
    assert sentences[1] == "It was introduced in 2017 by Google."
    assert sentences[2] == "What is it used for?"
    assert sentences[3] == "It is used for NLP!"

def test_split_into_sentences_abbreviations():
    text = "Dr. Vaswani et al. introduced the Transformer. It was published in Vol. 12, e.g., in a conference."
    sentences = split_into_sentences(text)
    
    # "Dr." and "et al." and "Vol." and "e.g." should not trigger splits
    assert len(sentences) == 2
    assert "Dr. Vaswani" in sentences[0]
    assert "Transformer." in sentences[0]

def test_semantic_chunker_basic():
    # Setup mock sentences
    sentences = [
        {"text": "Self-attention mechanisms are powerful.", "page": 1},
        {"text": "They allow matching sequences dynamically.", "page": 1},
        {"text": "FastAPI is a modern web framework.", "page": 2},
        {"text": "It is built on starlette and pydantic.", "page": 2}
    ]
    
    # We use a threshold percentile that forces splits, or mock embeddings
    # To run a fast test without loading sentence-transformers, we can mock the model property
    chunker = SemanticChunker()
    
    # Mocking the encoder to return high similarity for 0-1, low for 1-2, high for 2-3
    import numpy as np
    class MockModel:
        def encode(self, texts, show_progress_bar=False):
            # 384 dimensions
            vectors = np.zeros((len(texts), 384))
            # Topic A vectors
            vectors[0, :10] = 1.0
            vectors[1, :10] = 0.95
            # Topic B vectors
            vectors[2, 10:20] = 1.0
            vectors[3, 10:20] = 0.95
            return vectors
            
    chunker._model = MockModel()
    
    chunks = chunker.chunk_sentences(sentences)
    
    assert len(chunks) == 2
    # Verify groupings
    assert "Self-attention" in chunks[0]["text"]
    assert "matching sequences" in chunks[0]["text"]
    assert chunks[0]["pages"] == [1]
    
    assert "FastAPI" in chunks[1]["text"]
    assert "starlette" in chunks[1]["text"]
    assert chunks[1]["pages"] == [2]
