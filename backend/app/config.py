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
