CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

BASE_DIR = Path(__file__).resolve().parent

# Data storage
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
