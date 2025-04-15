import numpy as np
import os
import json
import pickle
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import re

from ..config import DATA_DIR, EMBEDDING_MODEL

class LocalVectorStore:
    """
    Local vector store implementation for storing and searching document embeddings.
    No external vector database dependencies.
    """
    def __init__(self, embedding_model_name: str):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index_path = os.path.join(DATA_DIR, "vector_index.pkl")
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
        # Load existing index if available
        self.load_index()
    
    def load_index(self):
        """Load index from disk if it exists."""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.embeddings = data.get('embeddings', [])
                    self.metadata = data.get('metadata', [])
                print(f"Loaded {len(self.documents)} documents from index")
            except Exception as e:
                print(f"Error loading index: {e}")
                # Initialize empty if loading fails
                self.documents = []
                self.embeddings = []
                self.metadata = []
    
    def save_index(self):
        """Save index to disk."""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadata': self.metadata
        }
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(data, f)
    
    def add_texts(self, texts: List[str], metadatas: List[Dict] = None):
        """
        Add texts to the vector store.
        
        Args:
            texts: List of text chunks to add
            metadatas: List of metadata dicts for each chunk
        """
        if not texts:
            return
        
        # Generate embeddings for the texts
        new_embeddings = self.embedding_model.encode(texts)
        
        # Add to our local storage
        self.documents.extend(texts)
        self.embeddings.extend(new_embeddings)
        
        # Add metadata
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{} for _ in texts])
        
        # Save the updated index
        self.save_index()
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, Dict, float]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (document, metadata, score) tuples
        """
        if not self.embeddings:
            return []
        
        # Convert list of embeddings to numpy array for efficient computation
        embeddings_array = np.array(self.embeddings)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate cosine similarity
        similarities = np.dot(embeddings_array, query_embedding) / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Return documents and scores
        results = []
        for idx in top_indices:
            results.append((
                self.documents[idx],
                self.metadata[idx],
                float(similarities[idx])
            ))
        
        return results
    
    def delete_by_metadata(self, metadata_field: str, value: Any) -> int:
        """
        Delete documents by metadata field.
        
        Args:
            metadata_field: Field to filter on
            value: Value to match
            
        Returns:
            Number of documents deleted
        """
        if not self.metadata:
            return 0
        
        # Find indices to delete
        indices_to_delete = []
        for i, meta in enumerate(self.metadata):
            if meta.get(metadata_field) == value:
                indices_to_delete.append(i)
        
        if not indices_to_delete:
            return 0
        
        # Create new lists excluding deleted indices
        new_documents = []
        new_embeddings = []
        new_metadata = []
        
        for i in range(len(self.documents)):
            if i not in indices_to_delete:
                new_documents.append(self.documents[i])
                new_embeddings.append(self.embeddings[i])
                new_metadata.append(self.metadata[i])
        
        # Update lists
        self.documents = new_documents
        self.embeddings = new_embeddings
        self.metadata = new_metadata
        
        # Save changes
        self.save_index()
        
        return len(indices_to_delete)
    
    def get_all_documents(self) -> List[Tuple[str, Dict]]:
        """Return all documents with their metadata."""
        return list(zip(self.documents, self.metadata))


# Initialize local vector store
vector_store = LocalVectorStore(EMBEDDING_MODEL)


class BM25:
    """
    Implementation of BM25 (Okapi BM25) algorithm for keyword search.
    """
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.corpus = []
    
    def tokenize(self, text):
        """Simple tokenization by splitting on non-word characters."""
        return re.findall(r'\w+', text.lower())
    
    def fit(self, corpus):
        """Fit the BM25 model on a corpus of documents."""
        self.corpus = [self.tokenize(doc) for doc in corpus]
        self.corpus_size = len(self.corpus)
        
        # Count document frequencies
        df = {}
        for document in self.corpus:
            tokens = set(document)
            for token in tokens:
                if token not in df:
                    df[token] = 0
                df[token] += 1
        
        # Compute inverse document frequencies
        self.idf = {}
        for token, freq in df.items():
            self.idf[token] = np.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)
        
        # Compute document lengths
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avg_doc_len = sum(self.doc_len) / self.corpus_size
    
    def score(self, query):
        """Score documents against a query using BM25."""
        query_tokens = self.tokenize(query)
        scores = np.zeros(self.corpus_size)
        
        for token in query_tokens:
            if token not in self.idf:
                continue
                
            # Calculate scores for each document
            for i, doc in enumerate(self.corpus):
                if token not in doc:
                    continue
                    
                # Count occurrences of token in document
                term_freq = doc.count(token)
                
                # BM25 scoring formula
                numerator = self.idf[token] * term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_doc_len)
                scores[i] += numerator / denominator
        
        return scores


def hybrid_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Implement a hybrid search combining keyword-based BM25 and semantic search.
    
    Args:
        query: User query
        top_k: Number of results to return
    
    Returns:
        List of search results with documents and metadata
    """
    # Get all documents for BM25
    all_docs = vector_store.get_all_documents()
    
    if not all_docs:
        return []
    
    # Extract texts and metadata
    texts = [doc[0] for doc in all_docs]
    metadatas = [doc[1] for doc in all_docs]
    
    # Initialize and fit BM25
    bm25 = BM25()
    bm25.fit(texts)
    
    # Get BM25 scores
    bm25_scores = bm25.score(query)
    
    # Get top 2*top_k candidates from BM25
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k*2]
    
    # Perform semantic search
    semantic_results = vector_store.similarity_search(query=query, k=top_k)
    
    # Extract document indices from semantic search results
    semantic_indices = []
    for i, (doc, _, _) in enumerate(semantic_results):
        try:
            semantic_indices.append(texts.index(doc))
        except ValueError:
            continue
    
    # Combine results using reciprocal rank fusion
    combined_results = {}
    
    # Score BM25 results
    for rank, idx in enumerate(bm25_top_indices):
        doc_id = str(idx)  # Use index as ID
        if doc_id not in combined_results:
            combined_results[doc_id] = {
                'document': texts[idx],
                'metadata': metadatas[idx],
                'score': 1/(rank + 60)  # RRF constant
            }
    
    # Score semantic results
    for rank, (doc, metadata, score) in enumerate(semantic_results):
        try:
            doc_id = str(texts.index(doc))
            if doc_id in combined_results:
                combined_results[doc_id]['score'] += 1/(rank + 60)  # RRF constant
            else:
                combined_results[doc_id] = {
                    'document': doc,
                    'metadata': metadata,
                    'score': 1/(rank + 60)  # RRF constant
                }
        except ValueError:
            continue
    
    # Sort by final score
    results = list(combined_results.values())
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results[:top_k]