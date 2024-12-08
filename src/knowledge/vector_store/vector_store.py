#src/knowledge/vector_store/vector_store.py
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        self.texts = []
        self.metadatas = []
        self.vector_store = None
        self._initialize_store()

    def _initialize_store(self) -> None:
        try:
            # Initialize with empty store
            self.vector_store = FAISS.from_texts(
                texts=[""],
                embedding=self.embeddings
            )
            # Clear initialization text
            if hasattr(self.vector_store, 'index'):
                self.vector_store.index.reset()
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        try:
            if not documents:
                return

            texts = [doc['content'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]

            # Store texts and metadata
            self.texts.extend(texts)
            self.metadatas.extend(metadatas)

            # Add to vector store
            if not self.vector_store:
                self.vector_store = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas
                )
            else:
                self.vector_store.add_texts(
                    texts=texts,
                    metadatas=metadatas
                )

            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    async def similarity_search(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            if not self.vector_store or not self.texts:
                return []

            # Use direct similarity search
            docs = self.vector_store.similarity_search(
                query=query,
                k=k
            )

            results = []
            for doc in docs:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': 1.0
                })

            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []