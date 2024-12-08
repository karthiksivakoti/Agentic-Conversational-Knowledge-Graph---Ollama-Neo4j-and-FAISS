#src/ingestion/pipeline_manager.py
from typing import List, Dict, Any, Optional
import asyncio
import logging
from pathlib import Path
from .processors.document_processors import TextSplitterProcessor, EntityExtractorProcessor
from .processors.embedding_processor import EmbeddingProcessor
from .processors.text_parser import TextParser
from src.knowledge.vector_store.vector_store import VectorStoreManager
from src.knowledge.graph_builder.graph_manager import KnowledgeGraphManager
from .processors.base_processor import Document

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """Manages the document ingestion pipeline"""
    
    def __init__(self, config: Dict[str, Any],
                 vector_store: Optional[VectorStoreManager] = None,
                 graph_manager: Optional[KnowledgeGraphManager] = None):
        self.config = config
        self.processors = self._initialize_processors()
        self.vector_store = vector_store or VectorStoreManager(config)
        self.graph_manager = graph_manager or KnowledgeGraphManager(config)
        logger.info("Initialized IngestionPipeline with all components")

    def _initialize_processors(self):
        """Initialize all document processors"""
        return [
            TextParser(self.config),
            TextSplitterProcessor(self.config),
            EmbeddingProcessor(self.config)
        ]

    async def process_document(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Process a single document through the pipeline"""
        try:
            # Create initial document
            doc = Document(
                content=content,
                metadata=metadata,
                doc_id=metadata.get('source', str(hash(content))),
                doc_type=metadata.get('type', 'unknown')
            )
            logger.info(f"Processing document: {doc.doc_id}")

            # Process through each processor
            current_docs = [doc]
            for processor in self.processors:
                try:
                    next_docs = []
                    for doc in current_docs:
                        if await processor.can_process(doc):
                            processed = await processor.process(doc)
                            next_docs.extend(processed)
                        else:
                            next_docs.append(doc)
                    current_docs = next_docs
                    logger.debug(f"Processed through {processor.__class__.__name__}")
                except Exception as e:
                    logger.error(f"Error in processor {processor.__class__.__name__}: {e}")
                    raise

            # Store in vector store
            logger.info("Storing documents in vector store")
            await self.vector_store.add_documents([{
                'content': doc.content,
                'metadata': doc.metadata
            } for doc in current_docs])

            # Store in graph database
            logger.info("Storing documents in graph database")
            for doc in current_docs:
                await self.graph_manager.add_document(doc)

            logger.info(f"Successfully processed document into {len(current_docs)} chunks")
            return current_docs

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise