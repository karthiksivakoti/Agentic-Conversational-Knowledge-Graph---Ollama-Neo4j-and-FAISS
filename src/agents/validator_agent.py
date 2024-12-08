#src/agents/validator_agent.py
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentState
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import asyncio
import json

class ValidationResult(BaseModel):
    is_valid: bool
    confidence_score: float
    reasoning: str
    improvements: List[str] = []
    sources: List[str] = []
    explanation: Dict[str, Any] = {}

class ValidatorAgentState(AgentState):
    context: Dict[str, Any] = {}
    test_results: List[Dict[str, Any]] = []
    validation_results: List[ValidationResult] = []
    final_explanation: Optional[Dict[str, Any]] = None

class ValidatorAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm = Ollama(model=config['llm']['model_name'])
        self.state = ValidatorAgentState()
        
        self.validation_prompt = PromptTemplate(
            template="""Validate the following response against the context and test results:
            
            Context: {context}
            Response: {response}
            Test Results: {test_results}
            
            Evaluate:
            1. Accuracy and completeness
            2. Logical consistency
            3. Source attribution
            4. Test coverage
            
            Provide:
            - Validation status
            - Confidence score
            - Detailed reasoning
            - Suggested improvements
            - Source references
            - Explanation for user
            
            Format as JSON.""",
            input_variables=["context", "response", "test_results"]
        )

    async def step(self, state: ValidatorAgentState) -> ValidatorAgentState:
        if state.current_step == 0:
            # Validate against test results
            validation = await self._validate_response(
                state.context.get('response', {}),
                state.context.get('test_results', [])
            )
            state.validation_results.append(validation)
            
        elif state.current_step == 1:
            # Generate explanation
            state.final_explanation = await self._generate_explanation(
                state.validation_results[0]
            )
            
        return state

    async def should_continue(self, state: ValidatorAgentState) -> bool:
        return state.current_step < 2

    async def _validate_response(
        self,
        response: Dict[str, Any],
        test_results: List[Dict[str, Any]]
    ) -> ValidationResult:
        """Validate response against test results and context"""
        try:
            validation_response = await self.llm.agenerate([
                self.validation_prompt.format(
                    context=json.dumps(self.state.context),
                    response=json.dumps(response),
                    test_results=json.dumps(test_results)
                )
            ])
            
            validation_data = json.loads(validation_response.generations[0].text)
            return ValidationResult(**validation_data)
            
        except Exception as e:
            print(f"Error in validation: {e}")
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                reasoning="Validation failed due to error",
                improvements=["Retry validation"],
                sources=[]
            )

    async def _generate_explanation(
        self,
        validation_result: ValidationResult
    ) -> Dict[str, Any]:
        """Generate user-friendly explanation of validation results"""
        explanation_prompt = PromptTemplate(
            template="""Generate a user-friendly explanation of the validation results:
            
            Validation Results: {validation_results}
            
            Include:
            1. Overall assessment
            2. Key findings
            3. Confidence level
            4. Areas for improvement
            5. Supporting evidence
            
            Make it clear and concise.""",
            input_variables=["validation_results"]
        )
        
        try:
            explanation_response = await self.llm.agenerate([
                explanation_prompt.format(
                    validation_results=validation_result.json()
                )
            ])
            
            explanation = {
                'summary': explanation_response.generations[0].text,
                'validation_details': validation_result.dict(),
                'confidence_level': validation_result.confidence_score,
                'improvement_suggestions': validation_result.improvements,
                'supporting_evidence': {
                    'sources': validation_result.sources,
                    'reasoning': validation_result.reasoning
                }
            }
            
            return explanation
            
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return {
                'summary': "Error generating explanation",
                'error': str(e)
            }

    def get_explanation(self) -> Optional[Dict[str, Any]]:
        """Get the final explanation for the user"""
        return self.state.final_explanation