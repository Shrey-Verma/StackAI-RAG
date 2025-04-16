import sys
sys.path.append("..")  # Add parent directory to path

from RAG.core.query_processing import query_classifier

print(f"Current threshold: {query_classifier.threshold}")
print(f"Feedback counts: {query_classifier.feedback_counts}")