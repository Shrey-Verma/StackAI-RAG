from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List
import os
import tempfile
import shutil

from api.models import (
    QueryRequest, 
    QueryResponse, 
    FileResponse, 
    FilesListResponse,
    DeleteFileRequest,
    DeleteFileResponse
)
from core.ingestion import ingest_file, delete_file, list_ingested_files
from core.query_processing import needs_document_search, transform_query
from core.search import hybrid_search
from core.reranking import rerank_results
from core.generation import prepare_context, generate_answer
from config import DATA_DIR

router = APIRouter()

@router.post("/ingest", response_model=FileResponse)
async def upload_files(file: UploadFile = File(...)):
    """Upload and ingest a PDF file."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    # Create a temporary file to store the uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # Copy the uploaded file to the temporary file
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Process the file
        file_id = ingest_file(temp_file_path, file.filename)
        
        # Return success response
        return FileResponse(file_id=file_id, filename=file.filename)
    finally:
        # Remove the temporary file
        os.unlink(temp_file_path)

@router.delete("/ingest", response_model=DeleteFileResponse)
async def delete_files(request: DeleteFileRequest):
    """Delete one or more ingested files."""
    deleted = []
    failed = []
    
    for file_id in request.file_ids:
        success = delete_file(file_id)
        if success:
            deleted.append(file_id)
        else:
            failed.append(file_id)
    
    return DeleteFileResponse(deleted=deleted, failed=failed)

@router.get("/ingest", response_model=FilesListResponse)
async def list_files():
    """List all ingested files."""
    files = list_ingested_files()
    response_files = [
        FileResponse(file_id=file['file_id'], filename=file['source'])
        for file in files
    ]
    
    return FilesListResponse(files=response_files)

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a user query and return a generated answer."""
    query = request.query
    
    # Check if the query needs document search
    if not needs_document_search(query):
        return QueryResponse(
            answer="I can answer this without searching the knowledge base.",
            sources=[]
        )
    
    # Transform the query
    transformed_query = transform_query(query)
    
    # Perform hybrid search
    search_results = hybrid_search(transformed_query, top_k=10)
    
    # Rerank results
    reranked_results = rerank_results(query, search_results, top_k=5)
    
    # Prepare context
    context = prepare_context(reranked_results)
    
    # Generate answer
    answer = generate_answer(query, context)
    
    # Extract sources
    sources = [result['metadata'].get('source', 'Unknown') for result in reranked_results]
    
    return QueryResponse(answer=answer, sources=sources)