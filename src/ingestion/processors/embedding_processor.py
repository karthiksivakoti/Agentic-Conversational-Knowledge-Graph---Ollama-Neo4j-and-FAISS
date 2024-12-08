#src/ingestion/processors/embedding_processor.py
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from .base_processor import BaseProcessor, Document

logger = logging.getLogger(__name__)

class EmbeddingProcessor(BaseProcessor):
    """Processor for generating document embeddings"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': self.device}
        )
        logger.info(f"Initialized EmbeddingProcessor with device: {self.device}")

    async def can_process(self, document: Document) -> bool:
        """Check if document can be processed"""
        return 'embedding' not in document.metadata

    async def process(self, document: Document) -> List[Document]:
        """Generate embeddings for document content"""
        try:
            # Generate embedding for document content
            embedding = self.embeddings.embed_documents([document.content])[0]
            
            # Update metadata with embedding
            metadata = document.metadata.copy()
            metadata['embedding'] = embedding  # embedding is already a list
            
            processed_doc = Document(
                content=document.content,
                metadata=metadata,
                doc_id=document.doc_id,
                doc_type=document.doc_type
            )
            
            logger.debug(f"Successfully generated embeddings for document: {document.doc_id}")
            return [processed_doc]
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise