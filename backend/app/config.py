import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
METADATA_STORE_PATH = DATA_DIR / "metadata_store.json"

# Load environment variables from root directory .env file
load_dotenv(dotenv_path=ROOT_DIR / ".env", override=True)
if (BASE_DIR / "app" / ".env").exists():
    load_dotenv(dotenv_path=BASE_DIR / "app" / ".env", override=True)


# Create directories if they do not exist
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Environment settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Pinecone Cloud settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "research-papers")

# Embedding settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")

# LLM settings
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
VERTEX_CREDENTIALS_PATH = os.getenv("VERTEX_CREDENTIALS_PATH")

# Auto-detect service.json relative to BASE_DIR (backend/), ROOT_DIR or current file CWD
default_service_json_backend = BASE_DIR / "service.json"
default_service_json_root = ROOT_DIR / "service.json"

if not VERTEX_CREDENTIALS_PATH:
    if default_service_json_backend.exists():
        VERTEX_CREDENTIALS_PATH = str(default_service_json_backend.resolve())
    elif default_service_json_root.exists():
        VERTEX_CREDENTIALS_PATH = str(default_service_json_root.resolve())
elif VERTEX_CREDENTIALS_PATH:
    # Resolve relative paths relative to ROOT_DIR
    resolved_path = ROOT_DIR / VERTEX_CREDENTIALS_PATH
    if not resolved_path.exists() and (BASE_DIR / VERTEX_CREDENTIALS_PATH).exists():
        resolved_path = BASE_DIR / VERTEX_CREDENTIALS_PATH
    VERTEX_CREDENTIALS_PATH = str(resolved_path.resolve())

if VERTEX_CREDENTIALS_PATH:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = VERTEX_CREDENTIALS_PATH




# AWS Bedrock settings
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-2-lite-v1:0")

# AWS S3 settings for RAG image ingestion
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "scholar-rag-images")


# OpenAI compatible Bedrock endpoint settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "openai.gpt-oss-120b")

def is_bedrock_configured() -> bool:
    return False

def get_bedrock_client():
    import boto3
    if not is_bedrock_configured():
        return None
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

def is_openai_configured() -> bool:
    return False

def get_openai_client():
    from openai import OpenAI
    if not is_openai_configured():
        return None
    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

# Langfuse settings
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")

LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


class PromptManager:
    def __init__(self, config_path: Path = CONFIG_DIR / "prompts.yaml"):
        self.config_path = config_path
        self.prompts = {}
        self.load_prompts()

    def load_prompts(self):
        if not self.config_path.exists():
            return
        with open(self.config_path, "r", encoding="utf-8") as f:
            try:
                self.prompts = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Error loading prompts YAML: {e}")
                self.prompts = {}

    def get_prompt(self, category: str = "rag_chat"):
        """Get the active version's system prompt and user prompt template."""
        self.load_prompts()  # Reload in case file changed (hot-reloading prompts)
        cat_config = self.prompts.get(category, {})
        active_version = cat_config.get("active_version", "v1")
        version_config = cat_config.get("versions", {}).get(active_version, {})
        
        system_prompt = version_config.get("system_prompt", "")
        user_template = version_config.get("user_prompt_template", "")
        return active_version, system_prompt, user_template

prompt_manager = PromptManager()
