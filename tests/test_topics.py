import sys
sys.path.append("..")

from core.query_processing import extract_document_topics

# Test data
sample_document = """
Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.

Recently, neural networks have become a cornerstone of machine learning approaches. Deep learning, a subset of machine learning, employs artificial neural networks with multiple layers to progressively extract higher-level features from raw input.

Python programming is widely used in the machine learning community due to its readability and vast ecosystem of libraries such as TensorFlow and PyTorch.
"""

# Extract topics
topics = extract_document_topics(sample_document)
print("Extracted topics:")
print(topics)