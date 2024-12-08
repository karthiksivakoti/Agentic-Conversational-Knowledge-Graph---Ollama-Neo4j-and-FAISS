#src/agents/query_agent.py
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentState
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from pydantic import BaseModel

class QueryType(BaseModel):
    type: str  # research, analysis, test_generation
    components: List[str]  # required knowledge components
    complexity: int  # 1-5 scale
    constraints: Dict[str, Any] = {}

class QueryAgentState(AgentState):
    query_type: QueryType = None
    parsed_query: Dict[str, Any] = {}
    required_context: List[str] = []

class QueryAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm = Ollama(model=config['llm']['model_name'])
        self.state = QueryAgentState()
        
        self.query_prompt = PromptTemplate(
            template="""Analyze the following query and break it down into components:
            Query: {query}
            
            Identify:
            1. Query type (research, analysis, test_generation)
            2. Required knowledge components
            3. Complexity level (1-5)
            4. Any specific constraints or requirements
            
            Provide your analysis in a structured format.""",
            input_variables=["query"]
        )

    async def step(self, state: QueryAgentState) -> QueryAgentState:
        if state.current_step == 0:
            # Initial query analysis
            response = await self.llm.agenerate([self.query_prompt.format(
                query=state.messages[0].content
            )])
            parsed_response = self._parse_llm_response(response.generations[0].text)
            state.query_type = QueryType(**parsed_response)
            
        elif state.current_step == 1:
            # Determine required context
            state.required_context = self._determine_required_context(state.query_type)
            
        return state

    async def should_continue(self, state: QueryAgentState) -> bool:
        return state.current_step < 2

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        # Add parsing logic here
        # This is a simplified version
        try:
            lines = response.strip().split('\n')
            parsed = {
                'type': 'research',  # Default
                'components': [],
                'complexity': 1,
                'constraints': {}
            }
            for line in lines:
                if 'type:' in line.lower():
                    parsed['type'] = line.split(':')[1].strip()
                elif 'components:' in line.lower():
                    parsed['components'] = [c.strip() for c in line.split(':')[1].split(',')]
                elif 'complexity:' in line.lower():
                    parsed['complexity'] = int(line.split(':')[1].strip())
            return parsed
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return parsed

    def _determine_required_context(self, query_type: QueryType) -> List[str]:
        """Determine what context needs to be fetched based on query type"""
        context_mapping = {
            'research': ['knowledge_graph', 'vector_store'],
            'analysis': ['vector_store', 'test_cases'],
            'test_generation': ['test_cases', 'knowledge_graph']
        }
        return context_mapping.get(query_type.type, ['vector_store'])