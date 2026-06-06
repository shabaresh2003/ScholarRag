import os
from google.oauth2 import service_account
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

# ==========================================
# 1. Authenticate using your service.json
# ==========================================
# Replace with the actual path to your service account key
SERVICE_ACCOUNT_FILE = "service.json" 
LOCATION = "us-central1" # Adjust to your GCP region

# Load credentials from the JSON file
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
PROJECT_ID = creds.project_id

print(f"Authenticated successfully to Project: {PROJECT_ID}")

# ==========================================
# 2. Initialize Vertex AI Models
# ==========================================
# Instantiate the Gemini LLM for generating critiques/scores
vertex_llm = ChatVertexAI(
    model_name="gemini-1.5-pro", # Or gemini-1.5-flash for faster/cheaper evals
    project=PROJECT_ID,
    location=LOCATION,
    credentials=creds,
    temperature=0.0 
)

# Instantiate the Embeddings model for metrics like Answer Relevancy
vertex_embeddings = VertexAIEmbeddings(
    model_name="text-embedding-004", 
    project=PROJECT_ID,
    location=LOCATION,
    credentials=creds
)

# Wrap them in Ragas wrappers so the framework understands them
ragas_llm = LangchainLLMWrapper(vertex_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(vertex_embeddings)

# ==========================================
# 3. Create Sample Evaluation Data
# ==========================================
# In production, this data comes from your application logs or test sets
data = {
    "question": [
        "What is Google Vertex AI?",
        "How do I optimize RAG?"
    ],
    "contexts": [
        ["Vertex AI is a fully managed, unified AI development platform by Google Cloud. It allows developers to train and deploy machine learning models quickly."],
        ["To optimize RAG, you can use advanced chunking strategies, fine-tune your embedding model, or implement GraphRAG."]
    ],
    "answer": [
        "Vertex AI is an AWS service for machine learning.", # Intentional hallucination to test the framework
        "You can optimize RAG by using better chunking, fine-tuning embeddings, and GraphRAG."
    ],
    "ground_truth": [
        "Vertex AI is a unified machine learning platform offered by Google Cloud.",
        "RAG can be optimized via chunking strategies, embedding fine-tuning, and using Graph databases."
    ]
}

# Convert dictionary to a HuggingFace Dataset, which Ragas requires
eval_dataset = Dataset.from_dict(data)

# ==========================================
# 4. Run the Ragas Evaluation
# ==========================================
print("Starting Ragas Evaluation...")

# Pass the metrics you want to evaluate
metrics_to_run = [
    faithfulness,       # Did the answer hallucinate beyond the context?
    answer_relevancy,   # Did the answer actually address the user's question?
    context_precision   # Were the right contexts retrieved?
]

result = evaluate(
    dataset=eval_dataset,
    metrics=metrics_to_run,
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    raise_exceptions=False # Prevents the whole run from failing if one row errors out
)

# ==========================================
# 5. Output the Results
# ==========================================
print("\n=== Overall Score ===")
print(result)

# Convert the results to a Pandas DataFrame for detailed inspection
df_results = result.to_pandas()
print("\n=== Detailed Row-by-Row Metrics ===")
print(df_results[['question', 'faithfulness', 'answer_relevancy']])