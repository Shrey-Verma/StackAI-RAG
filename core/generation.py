from typing import List, Dict, Any
import requests
import json
from config import MISTRAL_API_KEY

def prepare_context(results: List[Dict[str, Any]], max_tokens: int = 3000) -> str:
    """
    Prepare context from search results, limiting to max_tokens.
    
    Args:
        results: Reranked search results
        max_tokens: Maximum number of tokens for context
    
    Returns:
        Context string for the LLM prompt
    """
    context_parts = []
    current_tokens = 0
    token_estimate_factor = 1.3  # Rough estimate: 1 token ≈ 4 characters
    
    for result in results:
        # Estimate tokens for this document
        doc_text = result['document']
        doc_tokens = int(len(doc_text) / 4 * token_estimate_factor)
        
        # Check if adding this would exceed the limit
        if current_tokens + doc_tokens > max_tokens:
            # If we've already added at least one document, break
            if context_parts:
                break
            
            # Otherwise, truncate this document
            chars_to_include = int((max_tokens / token_estimate_factor) * 4)
            doc_text = doc_text[:chars_to_include] + "..."
        
        # Add source information
        source = result['metadata'].get('source', 'Unknown')
        context_parts.append(f"Source: {source}\n{doc_text}\n")
        
        # Update token count
        current_tokens += doc_tokens
    
    return "\n".join(context_parts)

def generate_answer(query: str, context: str) -> str:
    """
    Generate an answer using Mistral AI's API.
    
    Args:
        query: User query
        context: Context from search results
    
    Returns:
        Generated answer
    """
    if not MISTRAL_API_KEY:
        print("⚠️ MISTRAL_API_KEY is missing or empty!")
    else:
        print(f"✅ Using API key starting with: {MISTRAL_API_KEY[:4]}")

    # API endpoint
    url = "https://api.mistral.ai/v1/chat/completions"
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }
    
    # System prompt and messages
    system_prompt = "You are a helpful assistant answering questions based on the provided documents."
    prompt = f"""Answer the question based ONLY on the following context:

{context}

Question: {query}

If the context doesn't contain relevant information to answer the question, 
say "I don't have enough information to answer this question." 
Always cite the source of your information.

Answer:"""
    
    # Request body
    payload = {
        "model": "mistral-large-latest",  # Can also be "mistral-medium" or "mistral-small"
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }
    
    try:
        # Make the API call
        print(f"Making request to Mistral API with key: {MISTRAL_API_KEY[:4]}...")
        response = requests.post(url, headers=headers, json=payload)
        print(f"Status code: {response.status_code}")
        print(f"Response content: {response.text[:500]}")  # Print first 500 chars of response
        
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        # Parse the response
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        # Handle any errors with more detail
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Response error details: {e.response.text}")
        return f"Error calling Mistral AI API: {str(e)}"
