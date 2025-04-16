# RAG Backend System

A simple yet powerful Python backend for a Retrieval-Augmented Generation (RAG) pipeline, using a Language Model (LLM) for a knowledge base consisting of PDF files.

## System Design

This system enables users to upload PDF documents, query them using natural language, and receive AI-generated responses based on the relevant content from those documents. The backend is built using FastAPI and implements a custom RAG pipeline without relying on external RAG libraries.

## Architecture Overview

The system is divided into several core components:

1. **Data Ingestion** - Processes and stores PDF documents
2. **Query Processing** - Analyzes and transforms user queries
3. **Semantic Search** - Retrieves relevant document chunks
4. **Reranking** - Improves retrieval precision
5. **Generation** - Creates answers using an LLM

## Component Details

### 1. Data Ingestion

The ingestion process handles PDF uploads, text extraction, and chunking:

```python
def ingest_file(file_path: str, file_name: str) -> str:
    # Create a unique ID for the file
    file_id = hashlib.md5(file_name.encode()).hexdigest()
    
    # Extract text from PDF
    text = extract_text_from_pdf(file_path)
    
    # Extract topics for smart query detection
    topics = extract_document_topics(text)
    save_file_topics(file_id, topics)
    
    # Chunk text with overlap
    chunks = chunk_text(text, {"source": file_name, "file_id": file_id})
    
    # Store in vector database
    vector_store.add_texts([c[0] for c in chunks], [c[1] for c in chunks])
    
    return file_id
```

Key considerations for chunking:
- Chunks are created with overlap to preserve context
- Chunk boundaries respect paragraph structures where possible
- Metadata is preserved for each chunk, including source file information

### 2. Query Processing

The query processing pipeline uses a multi-stage approach to determine if a query needs document search:

```python
def needs_document_search(query: str) -> bool:
    # Stage 1: Rule-based filtering for obvious cases
    if matches_greeting_or_command_pattern(query):
        return False
    
    # Stage 2: Topic matching with document collection
    if query_contains_document_topics(query):
        return True
    
    # Stage 3: Lightweight ML classifier for ambiguous queries
    similarity = compare_to_conversational_queries(query)
    return similarity <= threshold  # Dynamic threshold
```

Query Processing Pipeline:
                  ┌─────────────────┐
                  │    User Query   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Intent Detection│
                  └────────┬────────┘
                           │
                           ▼
          ┌────────────────────────────┐
          │                            │
┌─────────▼─────────┐        ┌─────────▼─────────┐
│   Simple Intent   │        │ Knowledge-Based   │
│  (No Search Needed)│        │  (Search Needed)  │
└─────────┬─────────┘        └─────────┬─────────┘
          │                            │
          │                            ▼
          │                  ┌─────────────────┐
          │                  │Query Transformation│
          │                  └─────────┬────────┘
          │                            │
          │                            ▼
          │                  ┌─────────────────┐
          │                  │ Semantic Search │
          │                  └────────┬────────┘
          │                           │
┌─────────▼─────────┐        ┌────────▼────────┐
│  Direct Answer    │        │ Knowledge-Based │
│                   │        │    Response     │
└───────────────────┘        └─────────────────┘


For queries that need document retrieval, we transform them to improve search results:

```python
def transform_query(query: str) -> str:
    # Pattern-based transformations
    if matches_pattern("what is X"):
        return extract_topic(query) + " definition explanation information"
        
    if matches_pattern("how to X"):
        return "steps method process for " + extract_topic(query)
    
    # Return original if no patterns match
    return query
```

### 3. Semantic Search

The search component implements a hybrid approach combining keyword-based BM25 and semantic search:

```python
def hybrid_search(query: str, top_k: int = 5):
    # Get keyword matches with BM25
    bm25_results = bm25_search(query, top_k * 2)
    
    # Get semantic matches with embeddings
    semantic_results = vector_search(query, top_k)
    
    # Combine results with Reciprocal Rank Fusion
    combined_results = {}
    
    # Score and merge results
    for rank, result in enumerate(bm25_results):
        combined_results[result_id] = {
            "score": 1/(rank + 60) * result.score,
            "document": result.document,
            "metadata": result.metadata
        }
    
    for rank, result in enumerate(semantic_results):
        # Add or merge scores
        if result_id in combined_results:
            combined_results[result_id]["score"] += 1/(rank + 60) * result.score
        else:
            # Add new result
            # ...
    
    # Sort and return top results
    return sorted_results[:top_k]
```

### 4. Reranking

The reranking component improves result relevance and diversity:

```python
def rerank_results(query: str, results: List[Dict], top_k: int = 5):
    # First apply semantic reranking for relevance
    reranked = semantic_reranking(query, results)
    
    # Then apply Maximum Marginal Relevance for diversity
    final_results = apply_mmr(query, reranked, diversity=0.3, top_k=top_k)
    
    return final_results
```

### 5. Generation

The generation component uses Mistral AI to create answers based on the retrieved context:

```python
def generate_answer(query: str, context: str) -> str:
    # Prepare prompt
    prompt = f"""Answer the question based ONLY on the following context:

{context}

Question: {query}

If the context doesn't contain relevant information to answer the question, 
say "I don't have enough information to answer this question." 
Always cite the source of your information.

Answer:"""

    # Call Mistral AI API
    response = call_mistral_api(prompt)
    
    return response.content
```

## API Endpoints

The system provides the following RESTful endpoints:

1. **Ingestion Endpoints**
   - `POST /api/ingest` - Upload and process PDF files
   - `DELETE /api/ingest` - Delete ingested files
   - `GET /api/ingest` - List all ingested files
2. **Query Endpoint**
   - `POST /api/query` - Submit a query and get an AI-generated answer

## Technologies Used

### Core Libraries
- **FastAPI** - Web framework for building the API
- **PyMuPDF (fitz)** - PDF text extraction
- **Sentence-Transformers** - Text embeddings for semantic search
- **NumPy** - Numerical operations for search algorithms
- **Mistral AI API** - Language model for answer generation

## File Structure

```
stackai_rag/
├── app.py                  # FastAPI application entry point
├── config.py               # Configuration settings
├── requirements.txt        # Project dependencies
├── data/                   # Storage for ingested documents
├── api/
│   ├── __init__.py
│   ├── routes.py           # API endpoints definition
│   └── models.py           # Pydantic models for request/response
└── core/
    ├── __init__.py
    ├── ingestion.py        # Document ingestion functionality
    ├── query_processing.py # Query intent & transformation
    ├── search.py           # Vector search implementation
    ├── reranking.py        # Result reranking functionality
    └── generation.py       # LLM prompt generation
```

## Running the System

### Prerequisites
- Python 3.8 or higher
- Mistral AI API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Shrey-Verma/StackAI-RAG.git
   cd stackai_rag
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your Mistral AI API key:
   ```bash
   export MISTRAL_API_KEY=your_api_key_here
   ```

### Running the Server

Start the FastAPI server:
```bash
python app.py
```

The server will be accessible at http://localhost:8000, and the interactive API documentation will be available at http://localhost:8000/docs.

## Example Usage

1. Uploading a PDF document:
   ```bash
   curl -X POST http://localhost:8000/api/ingest \
     -F "file=@/path/to/your/document.pdf"
   ```

2. Querying the knowledge base:
   ```bash
   curl -X POST http://localhost:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is machine learning?"}'
   ```

## Algorithm Citations

1. **BM25 (Okapi BM25)** - Robertson, S. E., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4), 333-389.
2. **Maximum Marginal Relevance (MMR)** - Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. In Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 335-336).
3. **Reciprocal Rank Fusion** - Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. In Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval (pp. 758-759).
4. **Sentence-Transformers** - Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) (pp. 3982-3992).
