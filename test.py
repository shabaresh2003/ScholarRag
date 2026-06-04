import os
from google import genai

# 1. Point to your service account JSON key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service.json"

# 2. Set your Google Cloud Project ID and Region
PROJECT_ID = "ccme-genai"
LOCATION = "asia-south1" # e.g., us-central1, europe-west4, etc.

# 3. Initialize the unified GenAI client for Vertex AI
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

# 4. Generate content using Gemini 2.5 Flash
print("Calling Gemini 2.5 Flash...\n")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Wwhat is RAG"
)

print("--- Response ---")
print(response.text)