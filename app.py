from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from api.routes import router

app = FastAPI(
    title="RAG Backend API",
    description="Retrieval-Augmented Generation API for PDF documents",
    version="1.0.0"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

@app.get("/debug/intent")
async def debug_intent(query: str):
    """Debug endpoint to check query intent detection."""
    from core.query_processing import needs_document_search, transform_query
    
    needs_search = needs_document_search(query)
    transformed = transform_query(query) if needs_search else None
    
    return {
        "query": query,
        "needs_search": needs_search,
        "transformed": transformed
    }

@app.get("/debug/topics")
async def debug_topics():
    """Debug endpoint to check document topics."""
    import json
    from pathlib import Path
    from config import DATA_DIR
    
    topics_path = Path(DATA_DIR) / "document_topics.json"
    topics = []
    
    if topics_path.exists():
        with open(topics_path, 'r') as f:
            topics = json.load(f)
    
    return {
        "total_topics": len(topics),
        "topics": topics
    }

@app.get("/debug/search")
async def debug_search(query: str):
    """Debug endpoint to test search functionality."""
    from core.search import hybrid_search
    
    results = hybrid_search(query, top_k=3)
    
    return {
        "query": query,
        "results_count": len(results),
        "first_result": results[0] if results else None
    }

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "RAG Backend API"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)