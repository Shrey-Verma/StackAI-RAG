import os
import hashlib
import fitz  # PyMuPDF
from typing import List, Dict, Tuple
import re
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np

from config import DATA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from core.search import vector_store  # Import the local vector store

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(file_path)
    text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    
    doc.close()
    return text

class TextChunker:
    """
    Custom text chunker implementation to replace LangChain's RecursiveCharacterTextSplitter.
    Chunks text based on size with overlap.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks of approximately chunk_size characters with overlap.
        Tries to split on paragraph boundaries when possible.
        """
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            paragraph_size = len(paragraph)
            
            # If paragraph is already too big, split it further by sentences
            if paragraph_size > self.chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    sentence_size = len(sentence)
                    
                    # If adding this sentence would exceed chunk size, start a new chunk
                    if current_size + sentence_size > self.chunk_size and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        # Keep some overlap
                        overlap_size = 0
                        overlap_items = []
                        for item in reversed(current_chunk):
                            if overlap_size + len(item) <= self.chunk_overlap:
                                overlap_items.insert(0, item)
                                overlap_size += len(item) + 1  # +1 for space
                            else:
                                break
                        current_chunk = overlap_items
                        current_size = overlap_size
                    
                    current_chunk.append(sentence)
                    current_size += sentence_size + 1  # +1 for space
            else:
                # If adding this paragraph would exceed chunk size, start a new chunk
                if current_size + paragraph_size > self.chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Keep some overlap
                    overlap_size = 0
                    overlap_items = []
                    for item in reversed(current_chunk):
                        if overlap_size + len(item) <= self.chunk_overlap:
                            overlap_items.insert(0, item)
                            overlap_size += len(item) + 1  # +1 for space
                        else:
                            break
                    current_chunk = overlap_items
                    current_size = overlap_size
                
                current_chunk.append(paragraph)
                current_size += paragraph_size + 1  # +1 for space
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def create_documents(self, texts: List[str], metadatas: List[Dict] = None) -> List[Tuple[str, Dict]]:
        """Create document chunks with metadata."""
        if not metadatas:
            metadatas = [{}] * len(texts)
        
        all_chunks = []
        for i, text in enumerate(texts):
            chunks = self.split_text(text)
            for j, chunk in enumerate(chunks):
                # Create a copy of the metadata and add chunk info
                chunk_metadata = metadatas[i].copy()
                chunk_metadata['chunk'] = j
                all_chunks.append((chunk, chunk_metadata))
        
        return all_chunks

def chunk_text(text: str, metadata: Dict) -> List[Tuple[str, Dict]]:
    """Chunk text using custom text chunker."""
    text_splitter = TextChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    chunks = text_splitter.create_documents(
        [text], 
        metadatas=[metadata]
    )
    
    return chunks

def ingest_file(file_path: str, file_name: str) -> str:
    """Process a single file and add to vector store."""
    # Create a unique ID for the file
    file_id = hashlib.md5(file_name.encode()).hexdigest()
    
    # Extract text
    text = extract_text_from_pdf(file_path)
    
    # Create metadata
    metadata = {
        "source": file_name,
        "file_id": file_id
    }
    
    # Chunk text
    chunks = chunk_text(text, metadata)
    
    # Add to vector store
    texts = [chunk[0] for chunk in chunks]
    metadatas = [chunk[1] for chunk in chunks]
    
    vector_store.add_texts(texts=texts, metadatas=metadatas)
    
    return file_id

def delete_file(file_id: str) -> bool:
    """Delete all chunks associated with a file_id."""
    # Delete documents by metadata field
    deleted_count = vector_store.delete_by_metadata("file_id", file_id)
    return deleted_count > 0

def list_ingested_files() -> List[Dict]:
    """List all ingested files."""
    all_docs = vector_store.get_all_documents()
    
    # Extract unique files from metadata
    files = {}
    for doc, metadata in all_docs:
        file_id = metadata.get('file_id')
        if file_id and file_id not in files:
            files[file_id] = {
                'file_id': file_id,
                'source': metadata.get('source')
            }
    
    return list(files.values())