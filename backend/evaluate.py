import os
import sys
import json
import pandas as pd
from pathlib import Path
# ----------------------------------------------------
# Dynamic mock patch to fix deprecated langchain-community path required by Ragas
# ----------------------------------------------------
import sys
import types
try:
    from langchain_google_vertexai import ChatVertexAI
except ImportError:
    class ChatVertexAI:
        pass

# Create mock module
mock_vertex_module = types.ModuleType("langchain_community.chat_models.vertexai")
mock_vertex_module.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = mock_vertex_module

if "langchain_community.chat_models" not in sys.modules:
    mock_chat_models = types.ModuleType("langchain_community.chat_models")
    mock_chat_models.vertexai = mock_vertex_module
    sys.modules["langchain_community.chat_models"] = mock_chat_models
# ----------------------------------------------------

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Add parent directory to path to resolve imports
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from backend.app.database import get_db
from backend.app.pipeline import RAGPipeline
from backend.app.config import (
    GEMINI_API_KEY,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST,
    GEMINI_MODEL_NAME,
    is_bedrock_configured,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    BEDROCK_MODEL_ID
)

# Sample document context to seed database for evaluations
MOCK_PAPER_NAME = "transformer_evaluation_stub.pdf"
MOCK_CHUNKS = [
    {
        "text": "The Transformer is a deep learning model introduced in 2017. It is primarily used in natural language processing (NLP). The core architecture relies entirely on self-attention mechanisms, dispensing with recurrent neural networks (RNNs) or convolutional structures.",
        "pages": [1]
    },
    {
        "text": "Self-attention, sometimes called intra-attention, is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. It has been used successfully in reading comprehension and summarization.",
        "pages": [2]
    },
    {
        "text": "The multi-head attention mechanism allows the model to jointly attend to information from different representation subspaces at different positions. It splits queries, keys, and values into multiple heads, runs attention in parallel, and concatenates the outputs.",
        "pages": [3]
    }
]

# Evaluation QA dataset (Questions, Ground Truths)
GOLDEN_DATA = [
    {
        "question": "What is the core architecture of the Transformer model?",
        "ground_truth": "The core architecture of the Transformer model relies entirely on self-attention mechanisms, dispensing with recurrent neural networks (RNNs) or convolutional structures."
    },
    {
        "question": "What is self-attention and what is its alternative name?",
        "ground_truth": "Self-attention is also called intra-attention. It is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence."
    },
    {
        "question": "How does multi-head attention function?",
        "ground_truth": "Multi-head attention functions by splitting queries, keys, and values into multiple heads, executing attention in parallel across representation subspaces, and concatenating the outputs."
    }
]

def prepare_evaluation_db():
    """Ensures our database has at least the sample document indexed for evaluation."""
    print("Preparing evaluation database...")
    db = get_db()
    # Check if sample paper already indexed, otherwise add it
    if MOCK_PAPER_NAME not in db.get_all_document_names():
        print(f"Indexing mock paper '{MOCK_PAPER_NAME}' for evaluation...")
        db.add_document(MOCK_PAPER_NAME, MOCK_CHUNKS)
    else:
        print(f"Mock paper '{MOCK_PAPER_NAME}' is already indexed.")

def run_evaluation():
    if not is_bedrock_configured() and not GEMINI_API_KEY:
        print("ERROR: Either AWS Bedrock credentials or GEMINI_API_KEY must be set to run evaluations.")
        sys.exit(1)
        
    prepare_evaluation_db()
    
    print("\nRunning test queries through RAG pipeline...")
    pipeline = RAGPipeline()
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for item in GOLDEN_DATA:
        q = item["question"]
        gt = item["ground_truth"]
        print(f"Querying: '{q}'")
        
        # Run through pipeline
        try:
            # We filter specifically to our evaluation paper to run a clean assessment
            answer_obj, retrieved_docs = pipeline.run_query(q, source_filter=MOCK_PAPER_NAME)
            
            questions.append(q)
            answers.append(answer_obj.answer)
            # Ragas expects contexts as a list of strings
            contexts.append([doc["text"] for doc in retrieved_docs])
            ground_truths.append(gt)
        except Exception as e:
            print(f"Failed running query '{q}': {e}")
            
    if not questions:
        print("No queries completed successfully. Aborting evaluation.")
        return
        
    # Create evaluation dataset
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(dataset_dict)
    
    # Configure RAGAS metrics
    if is_bedrock_configured():
        print(f"\nConfiguring RAGAS to use AWS Bedrock model: {BEDROCK_MODEL_ID}...")
        from langchain_aws import ChatBedrock
        eval_llm = ChatBedrock(
            model_id=BEDROCK_MODEL_ID,
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            model_kwargs={"temperature": 0.0}
        )
        if GEMINI_API_KEY:
            os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
            eval_embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            )
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            eval_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        print("\nConfiguring RAGAS to use Gemini model...")
        # Inject Google API key for langchain classes
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
        
        eval_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME, 
            temperature=0.0,
            response_mime_type="application/json"
        )
        eval_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
    
    # Instantiate RAGAS metrics
    faithfulness_metric = Faithfulness(llm=eval_llm)
    answer_relevancy_metric = AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings)
    context_recall_metric = ContextRecall(llm=eval_llm)
    context_precision_metric = ContextPrecision(llm=eval_llm)
    
    metrics = [
        faithfulness_metric,
        answer_relevancy_metric,
        context_recall_metric,
        context_precision_metric
    ]
    
    # Initialize Langfuse tracing callback for evaluation run
    callbacks = []
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        try:
            from langfuse.langchain import CallbackHandler
            print("Langfuse keys found. Syncing evaluation traces to Langfuse...")
            handler = CallbackHandler(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST
            )
            callbacks.append(handler)
        except Exception as lf_err:
            print(f"Could not load Langfuse Callback Handler: {lf_err}")
            
    print("Evaluating with RAGAS (this may take a few moments)...")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            callbacks=callbacks
        )
        
        print("\n=== RAGAS EVALUATION METRICS ===")
        # Print results as clean pandas dataframe
        df = result.to_pandas()
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df)
        
        print("\n=== AVERAGE SCORES ===")
        mean_scores = df.mean(numeric_only=True)
        for metric_name, val in mean_scores.items():
            print(f"{metric_name.upper()}: {val:.4f}")
            
    except Exception as eval_err:
        print(f"RAGAS evaluation failed: {eval_err}")

if __name__ == "__main__":
    run_evaluation()
