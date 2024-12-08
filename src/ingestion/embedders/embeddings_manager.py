#src/ingesion/embedders/embeddings_manager.py
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

class EmbeddingsManager:
    """Manages document embeddings generation using HuggingFace models.
    
    This class provides a consistent interface for generating embeddings across our system,
    with automatic GPU acceleration when available."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize the embedding model with proper device configuration
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.get('vector_store', {}).get(
                'embedding_model', 
                'sentence-transformers/all-MiniLM-L6-v2'
            ),
            model_kwargs={'device': self.device}
        )

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to generate embeddings for
            
        Returns:
            List of embedding vectors (as float lists)"""
        return self.embedding_model.embed_documents(texts)

    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query text.
        
        Args:
            query: Query text to generate embedding for
            
        Returns:
            Query embedding vector"""
        return self.embedding_model.embed_query(query)