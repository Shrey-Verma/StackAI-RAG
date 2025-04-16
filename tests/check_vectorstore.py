import sys
sys.path.append("..")  # Add parent directory to path

from core.search import vector_store

# Check if vector store has documents
documents = vector_store.get_all_documents()
print(f"Vector store has {len(documents)} documents")
if not documents:
    print("WARNING: Vector store is empty!")