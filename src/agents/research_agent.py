#src/agents/research_agent.py
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentState
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.graphs import Neo4jGraph
from pydantic import BaseModel
import asyncio

class ResearchResult(BaseModel):
    sources: List[str]
    relevant_info: Dict[str, Any]
    confidence_score: float
    graph_relationships: List[Dict[str, Any]] = []

class ResearchAgentState(AgentState):
    query_context: Dict[str, Any] = {}
    vector_results: List[Dict[str, Any]] = []
    graph_results: List[Dict[str, Any]] = []
    final_results: Optional[ResearchResult] = None

class ResearchAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config['vector_db']['embedding_model']
        )
        self.vector_store = Chroma(
            persist_directory=config['vector_db']['persist_directory'],
            embedding_function=self.embeddings
        )
        self.graph_db = Neo4jGraph(
            url=config['graph_db']['uri'],
            username=config['graph_db']['username'],
            password=config['graph_db']['password']
        )
        self.state = ResearchAgentState()

    async def step(self, state: ResearchAgentState) -> ResearchAgentState:
        if state.current_step == 0:
            # Vector search
            state.vector_results = await self._perform_vector_search(
                state.query_context.get('query', '')
            )
        
        elif state.current_step == 1:
            # Graph search
            state.graph_results = await self._perform_graph_search(
                state.query_context.get('query', ''),
                state.vector_results
            )
        
        elif state.current_step == 2:
            # Synthesize results
            state.final_results = await self._synthesize_results(
                state.vector_results,
                state.graph_results
            )
        
        return state

    async def should_continue(self, state: ResearchAgentState) -> bool:
        return state.current_step < 3

    async def _perform_vector_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform semantic search using vector store"""
        try:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query,
                k=5
            )
            return [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                }
                for doc, score in results
            ]
        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    async def _perform_graph_search(
        self,
        query: str,
        vector_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Query knowledge graph based on vector search results"""
        try:
            # Extract relevant entities from vector results
            entities = self._extract_entities(vector_results)
            
            # Construct and execute Cypher query
            cypher_query = """
            MATCH (n)-[r]-(m)
            WHERE n.name IN $entities
            RETURN n, r, m
            LIMIT 10
            """
            results = self.graph_db.query(cypher_query, {'entities': entities})
            return results
        except Exception as e:
            print(f"Graph search error: {e}")
            return []

    async def _synthesize_results(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]]
    ) -> ResearchResult:
        """Combine and synthesize results from both searches"""
        # Combine relevant information
        relevant_info = {}
        sources = []
        relationships = []

        # Process vector results
        for result in vector_results:
            if result['score'] > 0.7:  # Confidence threshold
                sources.append(result['metadata'].get('source', 'unknown'))
                relevant_info[result['metadata'].get('id', 'unknown')] = result['content']

        # Process graph results
        for result in graph_results:
            relationships.append({
                'source': result.get('n', {}).get('name', ''),
                'relationship': result.get('r', {}).get('type', ''),
                'target': result.get('m', {}).get('name', '')
            })

        # Calculate confidence score
        confidence_score = sum(r['score'] for r in vector_results[:3]) / 3 if vector_results else 0.0

        return ResearchResult(
            sources=list(set(sources)),
            relevant_info=relevant_info,
            confidence_score=confidence_score,
            graph_relationships=relationships
        )

    def _extract_entities(self, vector_results: List[Dict[str, Any]]) -> List[str]:
        """Extract relevant entities from vector search results"""
        # Add entity extraction logic here
        # This is a simplified version
        entities = []
        for result in vector_results:
            if 'entities' in result['metadata']:
                entities.extend(result['metadata']['entities'])
        return list(set(entities))