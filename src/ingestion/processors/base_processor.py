#src/ingestion/processors/base_processor.py
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Document:
    """Base document class for all processors"""
    content: str
    metadata: Dict[str, Any]
    doc_id: str
    doc_type: str

class BaseProcessor(ABC):
    """Abstract base class for all document processors"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def process(self, document: Document) -> List[Document]:
        """Process a document and return processed documents"""
        pass

    @abstractmethod
    async def can_process(self, document: Document) -> bool:
        """Check if processor can handle this document"""
        pass