# tests/integration/test_rag_pipeline.py
import pytest
import asyncio
from pathlib import Path
import os
import logging
from typing import Dict, Any, List
from dataclasses import dataclass
import shutil

from src.ingestion.pipeline_manager import IngestionPipeline
from src.knowledge.graph_builder.graph_manager import KnowledgeGraphManager
from src.knowledge.indexer.hybrid_indexer import HybridIndexer
from src.knowledge.vector_store.vector_store import VectorStoreManager
from src.ingestion.loaders.document_loaders import DocumentLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestComponents:
    """Container for test components."""
    pipeline: IngestionPipeline
    indexer: HybridIndexer

@pytest.fixture(scope="function")
def clean_test_data():
    """Clean up test data before and after each test."""
    if os.path.exists('./test_data'):
        shutil.rmtree('./test_data')
    
    os.makedirs('./test_data', exist_ok=True)
    os.makedirs('./test_data/vector_store', exist_ok=True)
    os.makedirs('./test_data/parsing_results', exist_ok=True)
    
    yield
    
    if os.path.exists('./test_data'):
        shutil.rmtree('./test_data')

@pytest.fixture(scope="function")
def test_config():
    """Provide test configuration for each test function."""
    return {
        'vector_store': {
            'path': './test_data/vector_store',
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
        },
        'graph_db': {
            'uri': 'bolt://localhost:7687',
            'username': 'neo4j',
            'password': 'password123'
        },
        'llm': {
            'model_name': 'mistral',
            'temperature': 0.2,
            'max_tokens': 2000,
            'api_base': 'http://localhost:11434'
        }
    }

@pytest.fixture(scope="function")
def loaded_documents() -> List[Dict[str, Any]]:
    """Load documents synchronously for each test."""
    try:
        data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            logger.info(f"Created data directory: {data_dir}")

        documents = []
        for file_path in data_dir.glob('*.pdf'):
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(str(file_path))
                content = "".join(page.extract_text() for page in reader.pages)
                
                documents.append({
                    'content': content,
                    'metadata': {
                        'source': str(file_path),
                        'file_type': 'pdf',
                        'file_name': file_path.name
                    }
                })
                logger.info(f"Successfully loaded document: {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        if not documents:
            logger.warning("No documents found, using sample document")
            documents = [{
                'content': """Vehicle detection and classification are essential components 
                of modern traffic management systems. This document discusses the implementation
                of YOLOv11 for vehicle detection and ensemble OCR methods.""",
                'metadata': {
                    'source': 'sample_doc.txt',
                    'file_type': 'txt',
                    'file_name': 'sample_doc.txt'
                }
            }]

        logger.info(f"Loaded {len(documents)} documents:")
        for doc in documents:
            logger.info(f"\nDocument: {doc['metadata']['source']}")
            logger.info(f"Content length: {len(doc['content'])} characters")

        return documents

    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise

@pytest.fixture(scope="function")
def components(test_config, clean_test_data) -> TestComponents:
    """Create test components for each test function."""
    try:
        # Create fresh components for each test
        vector_store = VectorStoreManager(test_config)
        graph_manager = KnowledgeGraphManager(test_config)

        # Initialize components with fresh instances
        pipeline = IngestionPipeline(
            test_config,
            vector_store=vector_store,
            graph_manager=graph_manager
        )
        
        indexer = HybridIndexer(
            test_config,
            vector_store=vector_store,
            graph_manager=graph_manager
        )
        
        return TestComponents(pipeline=pipeline, indexer=indexer)
        
    except Exception as e:
        logger.error(f"Error in test setup: {e}")
        raise

@pytest.mark.asyncio
async def test_document_ingestion(components: TestComponents, loaded_documents: List[Dict[str, Any]]):
    """Test document ingestion and processing."""
    try:
        for doc in loaded_documents:
            logger.info(f"Processing document: {doc['metadata']['source']}")
            processed_docs = await components.pipeline.process_document(
                doc['content'], 
                doc['metadata']
            )
            
            # Verify basic processing
            assert len(processed_docs) > 0, "No documents were processed"
            assert all('embedding' in d.metadata for d in processed_docs), "Missing embeddings"
            assert all('entities' in d.metadata for d in processed_docs), "Missing entities"
            
            logger.info(f"Successfully processed document: {doc['metadata']['source']}")
            
    except Exception as e:
        logger.error(f"Document ingestion test failed: {e}")
        raise

@pytest.mark.asyncio
async def test_hybrid_search(components: TestComponents, loaded_documents: List[Dict[str, Any]]):
    """Test hybrid search functionality."""
    try:
        # Ingest documents again for this test
        for doc in loaded_documents:
            await components.pipeline.process_document(doc['content'], doc['metadata'])
            
        query = "What vehicle detection methods are discussed?"
        logger.info(f"Executing search query: {query}")
        
        results = await components.indexer.hybrid_search(query, k=2)
        
        assert len(results) > 0, "No search results found"
        assert all('content' in r for r in results), "Invalid result format"
        assert all('similarity_score' in r for r in results), "Missing relevance scores"
        
        logger.info(f"Search completed successfully with {len(results)} results")
        
    except Exception as e:
        logger.error(f"Hybrid search test failed: {e}")
        raise

@pytest.mark.asyncio
async def test_end_to_end_pipeline(components: TestComponents, loaded_documents: List[Dict[str, Any]]):
    """Test complete RAG pipeline end-to-end."""
    try:
        for doc in loaded_documents:
            logger.info(f"Processing document in end-to-end test: {doc['metadata']['source']}")
            processed_docs = await components.pipeline.process_document(
                doc['content'], 
                doc['metadata']
            )
            assert len(processed_docs) > 0, "Document processing failed"
            
        query = "Explain vehicle detection and classification methods"
        logger.info(f"Executing end-to-end test query: {query}")
        
        results = await components.indexer.hybrid_search(query)
        
        assert len(results) > 0, "No results returned in end-to-end test"
        
        logger.info("End-to-end pipeline test completed successfully")
        
    except Exception as e:
        logger.error(f"End-to-end pipeline test failed: {e}")
        raise

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--log-cli-level=INFO"])
