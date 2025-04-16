import sys
sys.path.append(".")

from RAG.core.query_processing import transform_query

# Test queries
test_queries = [
    "What is machine learning?",
    "How to train a neural network?",
    "Why use Python for AI?",
    "Explain the concept of deep learning"
]

print("Testing Query Transformation:")
print("----------------------------")
for query in test_queries:
    transformed = transform_query(query)
    print(f"Original:    '{query}'")
    print(f"Transformed: '{transformed}'")
    print()