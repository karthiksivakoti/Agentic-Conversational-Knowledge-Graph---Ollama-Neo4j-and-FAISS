#src/ingestion/processors/document_processors.py
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]
    doc_id: str
    doc_type: str

class BaseProcessor(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def process(self, document: Document) -> List[Document]:
        pass

    @abstractmethod
    async def can_process(self, document: Document) -> bool:
        pass

class TextSplitterProcessor(BaseProcessor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    async def can_process(self, document: Document) -> bool:
        return True

    async def process(self, document: Document) -> List[Document]:
        try:
            chunks = self.text_splitter.split_text(document.content)
            processed_docs = []
            
            for i, chunk in enumerate(chunks):
                doc_id = f"{document.doc_id}_chunk_{i}"
                metadata = document.metadata.copy()
                metadata['chunk_index'] = i
                metadata['total_chunks'] = len(chunks)
                
                processed_docs.append(Document(
                    content=chunk,
                    metadata=metadata,
                    doc_id=doc_id,
                    doc_type=document.doc_type
                ))
            
            return processed_docs
        except Exception as e:
            logger.error(f"Error in text splitting: {e}")
            return [document]

class EntityExtractorProcessor(BaseProcessor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.nlp = spacy.load("en_core_web_sm")

    async def can_process(self, document: Document) -> bool:
        return True

    async def process(self, document: Document) -> List[Document]:
        try:
            # If document already has entities, keep them
            if 'entities' in document.metadata:
                return [document]

            doc = self.nlp(document.content)
            entities = {
                ent.text: {
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                }
                for ent in doc.ents
            }

            # Update metadata with extracted entities
            metadata = document.metadata.copy()
            metadata['entities'] = entities

            return [Document(
                content=document.content,
                metadata=metadata,
                doc_id=document.doc_id,
                doc_type=document.doc_type
            )]
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return [document]