#src/agents/test_agent.py
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentState
from pydantic import BaseModel
import json
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import asyncio

class TestCase(BaseModel):
    id: str
    description: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    test_type: str  # unit, integration, edge_case
    importance: int  # 1-5 scale
    dependencies: List[str] = []
    validation_criteria: Dict[str, Any]

class TestAgentState(AgentState):
    context: Dict[str, Any] = {}
    generated_tests: List[TestCase] = []
    coverage_metrics: Dict[str, float] = {}

class TestAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm = Ollama(model=config['llm']['model_name'])
        self.state = TestAgentState()
        
        self.test_prompt = PromptTemplate(
            template="""Based on the following context and requirements, generate comprehensive test cases:
            
            Context: {context}
            Requirements: {requirements}
            
            Generate test cases that cover:
            1. Basic functionality
            2. Edge cases
            3. Error scenarios
            4. Integration points
            
            For each test case, provide:
            - Description
            - Input data
            - Expected output
            - Test type
            - Importance (1-5)
            - Dependencies
            - Validation criteria
            
            Format as JSON.""",
            input_variables=["context", "requirements"]
        )

    async def step(self, state: TestAgentState) -> TestAgentState:
        if state.current_step == 0:
            # Generate test cases
            test_cases = await self._generate_test_cases(state.context)
            state.generated_tests.extend(test_cases)
            
        elif state.current_step == 1:
            # Calculate coverage
            state.coverage_metrics = await self._calculate_coverage(
                state.generated_tests,
                state.context
            )
            
        elif state.current_step == 2:
            # Generate additional edge cases if needed
            if state.coverage_metrics.get('edge_case_coverage', 0) < 0.8:
                additional_tests = await self._generate_edge_cases(
                    state.context,
                    state.generated_tests
                )
                state.generated_tests.extend(additional_tests)
        
        return state

    async def should_continue(self, state: TestAgentState) -> bool:
        return state.current_step < 3

    async def _generate_test_cases(self, context: Dict[str, Any]) -> List[TestCase]:
        """Generate test cases based on context"""
        try:
            response = await self.llm.agenerate([
                self.test_prompt.format(
                    context=json.dumps(context.get('query_understanding', {})),
                    requirements=json.dumps(context.get('research', {}))
                )
            ])
            
            test_cases_data = json.loads(response.generations[0].text)
            return [TestCase(**test_case) for test_case in test_cases_data]
            
        except Exception as e:
            print(f"Error generating test cases: {e}")
            return []

    async def _calculate_coverage(
        self,
        test_cases: List[TestCase],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate test coverage metrics"""
        total_requirements = len(context.get('requirements', []))
        covered_requirements = set()
        
        # Coverage metrics
        coverage = {
            'overall_coverage': 0.0,
            'edge_case_coverage': 0.0,
            'integration_coverage': 0.0,
            'critical_path_coverage': 0.0
        }
        
        for test in test_cases:
            # Track covered requirements
            for dep in test.dependencies:
                covered_requirements.add(dep)
            
            # Calculate different coverage types
            if test.test_type == 'edge_case':
                coverage['edge_case_coverage'] += 1
            elif test.test_type == 'integration':
                coverage['integration_coverage'] += 1
            
            if test.importance >= 4:  # Critical path test
                coverage['critical_path_coverage'] += 1
        
        # Normalize coverage values
        total_tests = len(test_cases) or 1
        coverage['edge_case_coverage'] /= total_tests
        coverage['integration_coverage'] /= total_tests
        coverage['critical_path_coverage'] /= total_tests
        
        # Calculate overall coverage
        if total_requirements > 0:
            coverage['overall_coverage'] = len(covered_requirements) / total_requirements
        
        return coverage

    async def _generate_edge_cases(
        self,
        context: Dict[str, Any],
        existing_tests: List[TestCase]
    ) -> List[TestCase]:
        """Generate additional edge case tests"""
        edge_case_prompt = PromptTemplate(
            template="""Given the existing test cases and context, generate additional edge cases:
            
            Context: {context}
            Existing Tests: {existing_tests}
            
            Focus on:
            1. Boundary conditions
            2. Error scenarios
            3. Unexpected inputs
            4. Performance edge cases
            
            Format as JSON.""",
            input_variables=["context", "existing_tests"]
        )
        
        try:
            response = await self.llm.agenerate([
                edge_case_prompt.format(
                    context=json.dumps(context),
                    existing_tests=json.dumps([test.dict() for test in existing_tests])
                )
            ])
            
            test_cases_data = json.loads(response.generations[0].text)
            return [TestCase(**test_case) for test_case in test_cases_data]
            
        except Exception as e:
            print(f"Error generating edge cases: {e}")
            return []