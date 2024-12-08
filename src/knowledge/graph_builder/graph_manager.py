# src/knowledge/graph_builder/graph_manager.py
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver
from src.ingestion.loaders.base_loader import Document
import logging
import json

logger = logging.getLogger(__name__)

class KnowledgeGraphManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.driver: Optional[Driver] = None
        self._initialize_connection()
        self._setup_schema()
        self._clear_database()  # Clear database on initialization for testing

    def _initialize_connection(self):
        """Initialize Neo4j connection"""
        try:
            self.driver = GraphDatabase.driver(
                self.config['graph_db']['uri'],
                auth=(
                    self.config['graph_db']['username'],
                    self.config['graph_db']['password']
                )
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def _clear_database(self):
        """Clear all data from database"""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared Neo4j database")
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            raise

    def _setup_schema(self):
        """Set up initial schema constraints"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE (e.text, e.category) IS UNIQUE"
        ]

        try:
            with self.driver.session() as session:
                for constraint in constraints:
                    session.run(constraint)
                logger.info("Schema setup completed")
        except Exception as e:
            logger.error(f"Failed to setup schema: {e}")
            raise

    def _prepare_document_properties(self, document: Document) -> Dict[str, Any]:
        """Prepare document properties ensuring only primitive types"""
        return {
            'doc_id': document.doc_id,
            'content': document.content,
            'source': document.metadata.get('source', ''),
            'doc_type': document.metadata.get('type', 'unknown'),
            'summary': document.metadata.get('summary', '')
        }

    async def add_document(self, document: Document):
        """Add document and its entities to the graph"""
        try:
            # Prepare document properties
            doc_props = self._prepare_document_properties(document)

            # Create document node
            doc_query = """
            MERGE (d:Document {doc_id: $doc_id})
            SET d = $properties
            RETURN d
            """

            # Create/merge entity and relationship
            # Now we store category (label) of the entity as `category`
            entity_query = """
            MERGE (e:Entity {text: $text, category: $category})
            WITH e
            MATCH (d:Document {doc_id: $doc_id})
            MERGE (d)-[r:CONTAINS]->(e)
            SET r.start_pos = $start_pos,
                r.end_pos = $end_pos
            RETURN e, r, d
            """

            with self.driver.session() as session:
                # Add document
                session.run(doc_query, doc_id=doc_props['doc_id'], properties=doc_props)

                # Add entities and relationships
                entities = document.metadata.get('entities', {})
                for entity_key, entity_info in entities.items():
                    # entity_info has {text, label, start, end}
                    session.run(
                        entity_query,
                        text=entity_info['text'],
                        category=entity_info['label'],  # Storing label as category
                        doc_id=document.doc_id,
                        start_pos=entity_info['start'],
                        end_pos=entity_info['end']
                    )

            logger.info(
                f"Successfully added document {document.doc_id} with {len(entities)} entities"
            )

        except Exception as e:
            logger.error(f"Failed to add document to graph: {e}")
            raise

    async def get_entity_context(self, entity_text: str, category: str) -> Dict[str, Any]:
        """Get contextual information about an entity"""
        query = """
        MATCH (e:Entity {text: $entity_text, category: $category})
        OPTIONAL MATCH (d:Document)-[r:CONTAINS]->(e)
        WITH e, collect({
            doc_id: d.doc_id,
            content: d.content,
            start_pos: r.start_pos,
            end_pos: r.end_pos,
            doc_type: d.doc_type
        }) as documents
        RETURN {
            entity: properties(e),
            documents: documents
        } as context
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, entity_text=entity_text, category=category)
                record = result.single()
                if record:
                    return record['context']
                return {'entity': None, 'documents': []}
        except Exception as e:
            logger.error(f"Failed to get entity context: {e}")
            return {'entity': None, 'documents': []}

    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def __del__(self):
        self.close()
