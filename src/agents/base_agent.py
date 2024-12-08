#src/agents/base_agent.py
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from langchain.schema import BaseMessage
import langchain
from abc import ABC, abstractmethod

class AgentState(BaseModel):
    """Base state for all agents"""
    messages: List[BaseMessage] = []
    current_step: int = 0
    max_steps: int = 5
    metadata: Dict[str, Any] = {}

class BaseAgent(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = AgentState()
        self.name = self.__class__.__name__

    @abstractmethod
    async def step(self, state: AgentState) -> AgentState:
        """Execute one step of the agent's logic"""
        pass

    @abstractmethod
    async def should_continue(self, state: AgentState) -> bool:
        """Determine if the agent should continue processing"""
        pass

    def update_state(self, **kwargs) -> None:
        """Update the agent's state"""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)

    async def run(self, input_data: Any) -> AgentState:
        """Run the agent's full processing loop"""
        self.state.messages = []
        self.state.current_step = 0
        
        while (await self.should_continue(self.state)):
            self.state = await self.step(self.state)
            self.state.current_step += 1
            
        return self.state

    def get_state(self) -> AgentState:
        """Get current agent state"""
        return self.state