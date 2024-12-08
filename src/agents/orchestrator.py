#src/agents/orchestrator.py
from typing import Dict, List, Any, Optional
from langchain_community.graphs import Neo4jGraph  # Updated from GraphAPIWrapper
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
import asyncio
import logging

# Configure logging
logger = logging.getLogger(__name__)

class OrchestratorState(BaseModel):
    """Represents the current state of the orchestration process.
    
    This state object maintains the context and progress of the entire agent workflow,
    tracking which agent is currently active and storing intermediate results."""
    
    query: str
    context: Dict[str, Any] = {}
    current_agent: str = ""
    agent_states: Dict[str, Any] = {}
    final_response: Optional[str] = None
    error: Optional[str] = None

class AgentOrchestrator:
    """Manages the workflow between different agents in the RAG system.
    
    This orchestrator coordinates the interaction between various agents,
    maintaining the workflow sequence and ensuring proper data flow between components."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the orchestrator with configuration and setup graph database connection.
        
        Args:
            config: Configuration dictionary containing database and agent settings."""
        self.config = config
        self.agents = {}
        self.graph = StateGraph(OrchestratorState)
        
        # Initialize Neo4j connection for graph operations
        try:
            self.graph_db = Neo4jGraph(
                url=config['graph_db']['uri'],
                username=config['graph_db']['username'],
                password=config['graph_db']['password']
            )
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        
        self._setup_workflow()

    def register_agent(self, name: str, agent: Any) -> None:
        """Register an agent with the orchestrator.
        
        Args:
            name: Identifier for the agent
            agent: The agent instance to register"""
        if name in self.agents:
            logger.warning(f"Overwriting existing agent: {name}")
        self.agents[name] = agent
        logger.info(f"Successfully registered agent: {name}")

    def _setup_workflow(self) -> None:
        """Setup the workflow graph defining how agents interact.
        
        This method establishes the sequence of operations and the connections
        between different agents in the workflow."""
        try:
            # Define state transitions for each agent
            self.graph.add_node("query_understanding", self._run_query_agent)
            self.graph.add_node("research", self._run_research_agent)
            self.graph.add_node("test_generation", self._run_test_agent)
            self.graph.add_node("validation", self._run_validator_agent)

            # Define edges (workflow paths)
            self.graph.add_edge("query_understanding", "research")
            self.graph.add_edge("research", "test_generation")
            self.graph.add_edge("test_generation", "validation")
            self.graph.add_edge("validation", END)
            
            logger.info("Workflow graph setup completed successfully")
        except Exception as e:
            logger.error(f"Failed to setup workflow graph: {e}")
            raise

    async def _run_query_agent(self, state: OrchestratorState) -> OrchestratorState:
        """Execute query understanding agent to process initial user query.
        
        Args:
            state: Current orchestrator state
            
        Returns:
            Updated state with query understanding results"""
        try:
            agent = self.agents.get("query_agent")
            if agent:
                state.current_agent = "query_agent"
                logger.info("Starting query understanding phase")
                result = await agent.run(state.query)
                state.context.update({"query_understanding": result})
                logger.info("Query understanding completed successfully")
            else:
                logger.warning("Query agent not found")
        except Exception as e:
            logger.error(f"Error in query agent execution: {e}")
            state.error = str(e)
        return state

    async def _run_research_agent(self, state: OrchestratorState) -> OrchestratorState:
        """Execute research agent to gather relevant information.
        
        Args:
            state: Current orchestrator state
            
        Returns:
            Updated state with research results"""
        try:
            agent = self.agents.get("research_agent")
            if agent:
                state.current_agent = "research_agent"
                logger.info("Starting research phase")
                result = await agent.run(state.context)
                state.context.update({"research": result})
                logger.info("Research phase completed successfully")
            else:
                logger.warning("Research agent not found")
        except Exception as e:
            logger.error(f"Error in research agent execution: {e}")
            state.error = str(e)
        return state

    async def _run_test_agent(self, state: OrchestratorState) -> OrchestratorState:
        """Execute test generation agent to create validation tests.
        
        Args:
            state: Current orchestrator state
            
        Returns:
            Updated state with generated tests"""
        try:
            agent = self.agents.get("test_agent")
            if agent:
                state.current_agent = "test_agent"
                logger.info("Starting test generation phase")
                result = await agent.run(state.context)
                state.context.update({"tests": result})
                logger.info("Test generation completed successfully")
            else:
                logger.warning("Test agent not found")
        except Exception as e:
            logger.error(f"Error in test agent execution: {e}")
            state.error = str(e)
        return state

    async def _run_validator_agent(self, state: OrchestratorState) -> OrchestratorState:
        """Execute validation agent to verify results.
        
        Args:
            state: Current orchestrator state
            
        Returns:
            Updated state with validation results"""
        try:
            agent = self.agents.get("validator_agent")
            if agent:
                state.current_agent = "validator_agent"
                logger.info("Starting validation phase")
                result = await agent.run(state.context)
                state.final_response = result
                logger.info("Validation completed successfully")
            else:
                logger.warning("Validator agent not found")
        except Exception as e:
            logger.error(f"Error in validator agent execution: {e}")
            state.error = str(e)
        return state

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query through the complete agent workflow.
        
        Args:
            query: User's input query
            
        Returns:
            Dictionary containing response, context, and any errors"""
        try:
            logger.info(f"Processing new query: {query}")
            state = OrchestratorState(query=query)
            workflow = self.graph.compile()
            final_state = await workflow.arun(state)
            
            logger.info("Query processing completed successfully")
            return {
                "response": final_state.final_response,
                "context": final_state.context,
                "error": final_state.error
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"error": str(e)}

    def cleanup(self):
        """Cleanup resources and connections."""
        try:
            if hasattr(self, 'graph_db'):
                self.graph_db.close()
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")