from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer, util

class SimpleReranker:
    """
    A simple reranker that uses sentence transformers for reranking.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank search results using the model.
        
        Args:
            query: User query
            results: List of search results
            top_k: Number of results to return
        
        Returns:
            Reranked list of results
        """
        if not results:
            return []
        
        # Extract documents
        documents = [result['document'] for result in results]
        
        # Encode query and documents
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        doc_embeddings = self.model.encode(documents, convert_to_tensor=True)
        
        # Calculate similarity scores
        scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
        
        # Convert to numpy for easier handling
        scores = scores.cpu().numpy()
        
        # Add scores to results
        for i, score in enumerate(scores):
            results[i]['rerank_score'] = float(score)
        
        # Sort by rerank score
        reranked_results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        
        # Return top_k results
        return reranked_results[:top_k]


def apply_mmr(query: str, results: List[Dict[str, Any]], 
              model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
              diversity: float = 0.3, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Apply Maximum Marginal Relevance to rerank results for diversity.
    
    Args:
        query: User query
        results: List of search results
        model_name: Name of the embedding model
        diversity: Parameter controlling diversity (0-1)
        top_k: Number of results to return
    
    Returns:
        Reranked list of results for diversity
    """
    if len(results) <= 1:
        return results
    
    # Initialize model
    model = SentenceTransformer(model_name)
    
    # Extract documents and embed them
    documents = [result['document'] for result in results]
    document_embeddings = model.encode(documents)
    
    # Embed query
    query_embedding = model.encode(query)
    
    # Calculate similarities to query
    similarities = np.dot(document_embeddings, query_embedding) / (
        np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Initialize selected indices and remaining indices
    selected_indices = []
    remaining_indices = list(range(len(results)))
    
    # Select the first document with highest similarity to query
    first_idx = np.argmax(similarities)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Select the rest using MMR
    for _ in range(min(top_k - 1, len(results) - 1)):
        # Calculate MMR scores
        mmr_scores = np.zeros(len(remaining_indices))
        
        for i, idx in enumerate(remaining_indices):
            # Relevance term
            relevance = similarities[idx]
            
            # Diversity term
            diversity_term = 0
            for selected_idx in selected_indices:
                # Calculate similarity between this document and selected document
                sim = np.dot(document_embeddings[idx], document_embeddings[selected_idx]) / (
                    np.linalg.norm(document_embeddings[idx]) * np.linalg.norm(document_embeddings[selected_idx])
                )
                diversity_term = max(diversity_term, sim)
            
            # MMR score = relevance - diversity * max_similarity_to_selected
            mmr_scores[i] = (1 - diversity) * relevance - diversity * diversity_term
        
        # Select document with highest MMR score
        next_idx = remaining_indices[np.argmax(mmr_scores)]
        selected_indices.append(next_idx)
        remaining_indices.remove(next_idx)
    
    # Return selected documents in order of selection
    return [results[idx] for idx in selected_indices]


def rerank_results(query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Combine reranking approaches - first apply semantic reranking, then MMR.
    
    Args:
        query: User query
        results: List of search results
        top_k: Number of results to return
    
    Returns:
        Reranked list of results
    """
    if not results:
        return []
    
    # First apply semantic reranking
    reranker = SimpleReranker()
    semantically_reranked = reranker.rerank(query, results, top_k=min(top_k * 2, len(results)))
    
    # Then apply MMR for diversity
    final_results = apply_mmr(query, semantically_reranked, diversity=0.3, top_k=top_k)
    
    return final_results