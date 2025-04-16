import re
from typing import Dict, Tuple, List, Set
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
from pathlib import Path

from config import EMBEDDING_MODEL, DATA_DIR

# Path for storing feedback data and thresholds
FEEDBACK_PATH = Path(DATA_DIR) / "query_feedback.json"
TOPICS_PATH = Path(DATA_DIR) / "document_topics.json"

# Load embedding model for intent detection
model = SentenceTransformer(EMBEDDING_MODEL)

# Define common patterns that don't require document search
GREETING_PATTERNS = [
    r"^(hi|hello|hey|greetings|howdy)[\s\!\.\?]*$",
    r"^(good|hi|hello|hey) (morning|afternoon|evening|day)[\s\!\.\?]*$",
]

COMMAND_PATTERNS = [
    r"^(help|stop|quit|exit|cancel|clear)[\s\!\.\?]*$",
    r"^(thank you|thanks)[\s\!\.\?]*$",
    r"^(show|list) (documents|files|pdfs)[\s\!\.\?]*$",
    r"^(who are you|what can you do|tell me about yourself)[\s\!\.\?]*$",
]

# Define embeddings for common intents that don't need search
NO_SEARCH_EXAMPLES = [
    "How are you?",
    "What's your name?",
    "Who created you?",
    "What can you do?",
    "Tell me about yourself",
    "What are your capabilities?",
    "Can you help me?",
    "I need assistance",
    "Are you an AI?",
    "What time is it?",
]

# Pre-compute embeddings for no-search examples
no_search_embeddings = model.encode(NO_SEARCH_EXAMPLES)

class QueryIntentClassifier:
    """
    Multi-stage query intent classifier to determine if a document search is needed.
    Uses a combination of rule-based filtering, topic matching, and ML classification.
    """
    def __init__(self):
        self.threshold = 0.75  # Default threshold
        self.feedback_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        self.document_topics = set()
        self.load_data()
    
    def load_data(self):
        """Load saved thresholds, feedback, and document topics."""
        # Load threshold and feedback data
        if FEEDBACK_PATH.exists():
            try:
                with open(FEEDBACK_PATH, 'r') as f:
                    data = json.load(f)
                    self.threshold = data.get('threshold', 0.75)
                    self.feedback_counts = data.get('feedback_counts', self.feedback_counts)
            except Exception as e:
                print(f"Error loading feedback data: {e}")
        
        # Load document topics
        if TOPICS_PATH.exists():
            try:
                with open(TOPICS_PATH, 'r') as f:
                    self.document_topics = set(json.load(f))
            except Exception as e:
                print(f"Error loading document topics: {e}")
    
    def save_data(self):
        """Save threshold and feedback data."""
        os.makedirs(FEEDBACK_PATH.parent, exist_ok=True)
        with open(FEEDBACK_PATH, 'w') as f:
            json.dump({
                'threshold': self.threshold,
                'feedback_counts': self.feedback_counts
            }, f)
    
    def update_document_topics(self, topics: Set[str]):
        """Update the set of document topics."""
        self.document_topics.update(topics)
        os.makedirs(TOPICS_PATH.parent, exist_ok=True)
        with open(TOPICS_PATH, 'w') as f:
            json.dump(list(self.document_topics), f)
    
    def record_feedback(self, prediction: bool, actual: bool):
        """
        Record feedback to improve classification.
        
        Args:
            prediction: Whether the system predicted search was needed
            actual: Whether search was actually needed (user feedback)
        """
        if prediction and actual:
            self.feedback_counts["tp"] += 1
        elif prediction and not actual:
            self.feedback_counts["fp"] += 1
        elif not prediction and actual:
            self.feedback_counts["fn"] += 1
        else:  # not prediction and not actual
            self.feedback_counts["tn"] += 1
        
        # Update threshold dynamically based on feedback
        self._adjust_threshold()
        self.save_data()
    
    def _adjust_threshold(self):
        """Dynamically adjust the threshold based on feedback."""
        # Only adjust if we have enough data
        total = sum(self.feedback_counts.values())
        if total < 10:
            return
        
        # Calculate precision and recall
        tp = self.feedback_counts["tp"]
        fp = self.feedback_counts["fp"]
        fn = self.feedback_counts["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Adjust threshold based on precision vs recall balance
        # If too many false positives, increase threshold
        if precision < 0.7 and self.threshold < 0.9:
            self.threshold += 0.02
        # If too many false negatives, decrease threshold
        elif recall < 0.7 and self.threshold > 0.6:
            self.threshold -= 0.02
    
    def extract_topics(self, query: str) -> Set[str]:
        """Extract potential topics from query using basic NLP."""
        # Remove stop words and punctuation
        stop_words = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", 
            "with", "by", "about", "from", "of", "is", "are", "am", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did", "can",
            "could", "will", "would", "shall", "should", "may", "might", "must", "i",
            "you", "he", "she", "it", "we", "they", "me", "him", "her", "them", "us",
            "my", "your", "his", "its", "our", "their", "this", "that", "these", "those"
        }
        
        # Tokenize and clean
        query = re.sub(r'[^\w\s]', ' ', query.lower())
        words = query.split()
        
        # Filter out stop words
        topics = {word for word in words if word not in stop_words and len(word) > 2}
        
        return topics
    
    def needs_document_search(self, query: str) -> bool:
        """
        Multi-stage approach to determine if a query needs document search.
        
        1. Rule-based fast filtering
        2. Topic matching with document collection
        3. Embedding similarity for ambiguous cases
        """
        # Normalize query
        query = query.strip().lower()
        
        # Stage 1: Rule-based filtering (fast pre-filtering)
        for pattern in GREETING_PATTERNS + COMMAND_PATTERNS:
            if re.match(pattern, query):
                return False
        
        # Stage 2: Topic matching
        query_topics = self.extract_topics(query)
        # If query contains topics that match our document topics, likely needs search
        if query_topics and self.document_topics:
            for topic in query_topics:
                if topic in self.document_topics or any(topic in doc_topic for doc_topic in self.document_topics):
                    return True
        
        # Stage 3: Lightweight ML classifier for ambiguous queries
        query_embedding = model.encode([query])[0]
        
        # Compare with pre-computed embeddings
        similarities = [np.dot(query_embedding, emb) for emb in no_search_embeddings]
        max_similarity = max(similarities)
        
        # Use the dynamic threshold
        return max_similarity <= self.threshold


# Initialize the classifier
query_classifier = QueryIntentClassifier()

def needs_document_search(query: str) -> bool:
    """
    Determine if a query needs document search using the multi-stage classifier.
    
    Args:
        query: User query
        
    Returns:
        Boolean indicating if search is needed
    """
    return query_classifier.needs_document_search(query)

def record_search_feedback(query: str, prediction: bool, actual: bool):
    """
    Record feedback about search predictions.
    
    Args:
        query: The original query
        prediction: Whether system predicted search was needed
        actual: User feedback on whether search was actually needed
    """
    query_classifier.record_feedback(prediction, actual)

def update_document_topics(topics: Set[str]):
    """
    Update the document topics index.
    Should be called when new documents are added.
    
    Args:
        topics: New topics from documents
    """
    query_classifier.update_document_topics(topics)

def transform_query(query: str) -> str:
    """
    Transform the query to improve retrieval performance.
    Uses a template-based approach for common patterns.
    """
    # Convert to lowercase and strip whitespace
    query = query.strip()
    
    # Pattern 1: "What is X" -> "X definition explanation information"
    if re.match(r"^what (is|are) ", query.lower()):
        topic = re.sub(r"^what (is|are) ", "", query.lower())
        return f"{topic} definition explanation information"
    
    # Pattern 2: "How to X" -> "steps method process for X"
    if re.match(r"^how (to|do|can|could) ", query.lower()):
        topic = re.sub(r"^how (to|do|can|could) ", "", query.lower())
        return f"steps method process for {topic}"
    
    # Pattern 3: "Why X" -> "reasons explanation for X"
    if re.match(r"^why ", query.lower()):
        topic = re.sub(r"^why ", "", query.lower())
        return f"reasons explanation for {topic}"
    
    # Default: return original query
    return query

# Function to extract topics from documents - call this during document ingestion
def extract_document_topics(text: str) -> Set[str]:
    """
    Extract key topics from document text.
    
    Args:
        text: Document text
        
    Returns:
        Set of topic words
    """
    # Simple implementation - could be improved with TF-IDF or topic modeling
    stop_words = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", 
        "with", "by", "about", "from", "of", "is", "are", "am", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "can",
        "could", "will", "would", "shall", "should", "may", "might", "must"
    }
    
    # Tokenize and clean
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    
    # Filter out stop words and short words
    topics = {word for word in words if word not in stop_words and len(word) > 3}
    
    # Keep only the most common terms (to avoid too many topics)
    word_counts = {}
    for word in words:
        if word not in stop_words and len(word) > 3:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Get top 50 most frequent words
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return {word for word, _ in sorted_words[:50]}