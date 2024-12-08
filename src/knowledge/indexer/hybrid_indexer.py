# src/knowledge/indexer/hybrid_indexer.py
from typing import List, Dict, Any, Optional
import logging
import torch
from src.knowledge.vector_store.vector_store import VectorStoreManager
from src.knowledge.graph_builder.graph_manager import KnowledgeGraphManager

logger = logging.getLogger(__name__)

class HybridIndexer:
    """Combines vector-based and graph-based search capabilities"""

    def __init__(self, config: Dict[str, Any],
                 vector_store: Optional[VectorStoreManager] = None,
                 graph_manager: Optional[KnowledgeGraphManager] = None):
        self.config = config
        self.vector_store = vector_store or VectorStoreManager(config)
        self.graph_manager = graph_manager or KnowledgeGraphManager(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    async def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and graph context.
        Steps:
        1. Perform vector similarity search to find top-k relevant documents.
        2. For each result, extract entities and attempt to fetch graph context for at least one entity.
        3. Combine vector score and graph context presence to get a final score.
        4. Return top results sorted by combined score.
        """
        try:
            # Vector search
            vector_results = await self.vector_store.similarity_search(query, k=k)

            if not vector_results:
                logger.warning("No vector search results found")
                return []

            final_results = []
            seen_contents = set()

            for result in vector_results:
                if len(final_results) >= k:
                    break

                content = result['content']
                # Avoid duplicates
                if content in seen_contents:
                    continue
                seen_contents.add(content)

                # Attempt to get graph context
                graph_context = None
                entities = result.get('metadata', {}).get('entities', {})

                # If we have entities, try to fetch graph context for one of them
                # expecting each entity_info to have a 'label' field from parsing.
                if entities:
                    for entity_text, entity_info in entities.items():
                        category = entity_info.get('label', 'CONCEPT')
                        # Now we call get_entity_context with both entity_text and category
                        context = await self.graph_manager.get_entity_context(entity_text, category)
                        if context['entity'] is not None:
                            graph_context = context
                            break

                # Vector score is from result; default to 1.0 if not provided
                vector_score = result.get('similarity_score', 1.0)
                # If we got graph context, we boost score
                graph_score = 1.0 if graph_context else 0.0
                combined_score = (alpha * vector_score) + ((1 - alpha) * graph_score)

                final_results.append({
                    'content': content,
                    'metadata': result.get('metadata', {}),
                    'similarity_score': combined_score,
                    'graph_context': graph_context
                })

            # Sort by combined score
            final_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return final_results[:k]

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []

    def close(self):
        """Cleanup resources"""
        if hasattr(self, 'graph_manager'):
            self.graph_manager.close()
