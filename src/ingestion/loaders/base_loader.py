#src/ingestion/loaders/base_loader.py
from typing import List, Dict, Any, BinaryIO
from abc import ABC, abstractmethod
from pathlib import Path
import asyncio
from pydantic import BaseModel

class Document(BaseModel):
    content: str
    metadata: Dict[str, Any]
    doc_type: str
    doc_id: str

class BaseLoader(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_types: List[str] = []

    @abstractmethod
    async def load(self, source: Any) -> List[Document]:
        """Load document from source"""
        pass

    @abstractmethod
    async def can_handle(self, source: Any) -> bool:
        """Check if loader can handle this source"""
        pass

    async def validate(self, document: Document) -> bool:
        """Validate loaded document"""
        return (
            document.content is not None
            and document.doc_id is not None
            and document.doc_type in self.supported_types
        )

class BaseProcessor(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def process(self, document: Document) -> List[Document]:
        """Process a document"""
        pass

    @abstractmethod
    async def can_process(self, document: Document) -> bool:
        """Check if processor can handle this document"""
        pass