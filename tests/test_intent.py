import sys
# sys.path.append("..")

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add current directory to path

from RAG.core.query_processing import needs_document_search, update_document_topics, record_search_feedback

# Test data
test_queries = [
    # Greetings (should return False)
    "Hello there",
    "Hi, how are you?",
    "Good morning",
    
    # Commands (should return False)
    "Help me with something",
    "Thank you for your assistance",
    "Who are you?",
    
    # Ambiguous queries (check threshold)
    "What can you do for me?",
    "Are you smart?",
    
    # Document-related queries (should return True)
    "What is machine learning?",
    "How do neural networks work?",
    "Tell me about Python programming",
]

# Add some document topics
update_document_topics({"machine", "learning", "neural", "networks", "python", "programming"})

# Test queries
print("Testing Query Intent Detection:")
print("------------------------------")
for query in test_queries:
    result = needs_document_search(query)
    print(f"Query: '{query}'")
    print(f"Needs search: {result}")
    print()

# Test feedback loop
print("Testing Feedback Loop:")
print("---------------------")
test_query = "What is the weather today?"
prediction = needs_document_search(test_query)
print(f"Query: '{test_query}'")
print(f"Initial prediction: {prediction}")

# Provide feedback that this doesn't need search
record_search_feedback(test_query, prediction, False)
new_prediction = needs_document_search(test_query)
print(f"Prediction after feedback: {new_prediction}")