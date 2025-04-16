import sys
# sys.path.append("..")

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RAG.core.ingestion import ingest_file
from RAG.core.query_processing import needs_document_search, transform_query, extract_document_topics
from RAG.core.search import hybrid_search
from RAG.core.reranking import rerank_results
from RAG.core.generation import prepare_context, generate_answer

# Step 1: Load a test PDF
test_file_path = "/Users/heyshrey/Documents/Resume_ShreyVerma.pdf"  # Replace with actual path
file_name = "Resume_ShreyVerma.pdf"

print("1. Ingesting test document...")
file_id = ingest_file(test_file_path, file_name)
print(f"File ingested with ID: {file_id}")
print()

# Step 2: Test a query that should trigger document search
test_query = "What is his experience?"  # Replace with a query relevant to your test document

print("2. Testing query intent detection...")
needs_search = needs_document_search(test_query)
print(f"Query: '{test_query}'")
print(f"Needs search: {needs_search}")
print()

if needs_search:
    # Step 3: Transform the query
    print("3. Transforming query...")
    transformed_query = transform_query(test_query)
    print(f"Transformed: '{transformed_query}'")
    print()
    
    # Step 4: Perform hybrid search
    print("4. Performing hybrid search...")
    search_results = hybrid_search(transformed_query, top_k=5)
    print(f"Found {len(search_results)} results")
    print()
    
    # Step 5: Rerank results
    print("5. Reranking results...")
    reranked_results = rerank_results(test_query, search_results, top_k=3)
    print(f"Reranked to {len(reranked_results)} results")
    print()
    
    # Step 6: Prepare context
    print("6. Preparing context...")
    context = prepare_context(reranked_results)
    print(f"Context length: {len(context)} characters")
    print()
    
    # Step 7: Generate answer
    print("7. Generating answer...")
    answer = generate_answer(test_query, context)
    print(f"Answer: {answer}")
else:
    print("Query doesn't need document search")