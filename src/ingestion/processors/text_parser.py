# src/ingestion/processors/text_parser.py
from typing import Dict, Any, List
import logging
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from .base_processor import BaseProcessor, Document
import json

logger = logging.getLogger(__name__)

class Entity(BaseModel):
    text: str
    label: str  # e.g., TECHNOLOGY, METHOD, CONCEPT, METRIC, TERM
    start: int
    end: int

class ParsedContent(BaseModel):
    """Structure for parsed document content with richer metadata"""
    entities: Dict[str, Entity] = Field(description="Dictionary of extracted entities keyed by their text")
    document_type: str = Field(description="Type of document (e.g., research, technical, manual)")
    summary: str = Field(description="A brief but technical summary of the document")

class TextParser(BaseProcessor):
    """Improved intelligent text parser using LLM with richer entity categorization."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm = ChatOllama(
            model=config['llm'].get('model_name', 'mistral'),
            temperature=0.2,
            format="json"
        )
        self.output_parser = PydanticOutputParser(pydantic_object=ParsedContent)

        # Double braces for literal JSON
        schema_str = """
{{
  "entities": {{
    "<unique_entity_key>": {{
      "text": "exact text from document",
      "label": "TECHNOLOGY/METHOD/CONCEPT/METRIC/TERM",
      "start": 0,
      "end": 0
    }}
  }},
  "document_type": "one of: research, technical, manual, scholarly, report, etc.",
  "summary": "brief technical summary of the document"
}}
"""

        # GOOD RESPONSE EXAMPLE also with doubled braces
        # All braces inside JSON must be doubled
        good_example = """
{{
  "entities": {{
    "YOLOv11": {{
      "text": "YOLOv11",
      "label": "TECHNOLOGY",
      "start": 145,
      "end": 152
    }},
    "Ensemble OCR": {{
      "text": "Ensemble OCR",
      "label": "METHOD",
      "start": 200,
      "end": 212
    }}
  }},
  "document_type": "research",
  "summary": "A research document outlining YOLOv11-based vehicle detection and ensemble OCR techniques."
}}
"""

        # BAD RESPONSE EXAMPLE
        bad_example = """{{"Text": "some text", "Summary":"description"}}"""

        # No f-string here, just a normal string concatenation
        # Everything that needs to be literal braces is doubled.
        system_message = (
            "You are an expert at analyzing technical documents.\n"
            "Analyze the given text and extract key entities, assigning each entity a label from [TECHNOLOGY, METHOD, CONCEPT, METRIC, TERM].\n"
            "Also classify the document_type (e.g., research, technical) and provide a technical summary.\n\n"
            "Respond strictly in this JSON format:\n"
            + schema_str +
            "\n\nGUIDELINES:\n"
            "1. The response MUST be valid JSON and match the exact schema.\n"
            "2. Entities should be meaningful and reflect actual concepts from the text.\n"
            "3. Choose the most suitable label from [TECHNOLOGY, METHOD, CONCEPT, METRIC, TERM].\n"
            "4. Keep the summary concise and technical.\n\n"
            "BAD RESPONSE EXAMPLE:\n"
            + bad_example +
            "\n\nGOOD RESPONSE EXAMPLE:\n"
            + good_example
        )

        self.parse_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "Text to analyze: {text}")
        ])

    async def preprocess_content(self, content: str) -> str:
        """Clean and prepare content for parsing"""
        content = " ".join(content.split())
        max_length = 4000
        if len(content) > max_length:
            content = content[:max_length] + "..."
        return content

    async def can_process(self, document: Document) -> bool:
        """Check if document needs parsing"""
        return 'entities' not in document.metadata

    async def process(self, document: Document) -> List[Document]:
        """Parse document content using LLM with improved entity extraction."""
        try:
            processed_content = await self.preprocess_content(document.content)
            logger.debug(f"Processing document: {document.doc_id} ({len(processed_content)} chars)")

            messages = self.parse_prompt.format_messages(text=processed_content)

            max_retries = 3
            parsed_content = None
            for attempt in range(max_retries):
                try:
                    response = await self.llm.ainvoke(messages)
                    parsed_data = json.loads(response.content)

                    # Validate required fields
                    required_fields = {'entities', 'document_type', 'summary'}
                    if not all(field in parsed_data for field in required_fields):
                        missing = required_fields - set(parsed_data.keys())
                        raise ValueError(f"Missing required fields: {missing}")

                    parsed_content = ParsedContent(**parsed_data)
                    logger.info(
                        f"Successfully parsed document {document.doc_id} "
                        f"with {len(parsed_content.entities)} entities"
                    )
                    break
                except (json.JSONDecodeError, ValueError) as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    continue

            if parsed_content is None:
                raise RuntimeError("Parsing failed after all retries.")

            metadata = document.metadata.copy()
            # Update metadata with richer entity info and refined document_type
            metadata.update({
                'entities': {k: v.dict() for k, v in parsed_content.entities.items()},
                'type': parsed_content.document_type,
                'summary': parsed_content.summary
            })

            processed_doc = Document(
                content=document.content,
                metadata=metadata,
                doc_id=document.doc_id,
                doc_type=parsed_content.document_type
            )

            return [processed_doc]

        except Exception as e:
            logger.error(f"Error parsing document: {e}")
            raise
