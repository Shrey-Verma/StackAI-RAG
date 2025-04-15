import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data storage
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"  # Renamed from VECTOR_DB_DIR

# Make sure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# LLM settings - Mistral AI
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "9C93VQatItAiSncxwjle0iPZda0uqvVp")
LLM_MODEL = "mistral-large-latest"  # You can also use "mistral-medium" or "mistral-small"