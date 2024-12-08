#config/config.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

class LLMConfig(BaseModel):
    model_name: str = "mistral:7b"
    temperature: float = 0.7
    max_tokens: int = 2000
    api_base: Optional[str] = "http://localhost:11434"
    
class VectorDBConfig(BaseModel):
    db_type: str = "chroma"
    collection_name: str = "rag_collection"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    persist_directory: str = "./data/vectordb"

class GraphDBConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"

class AgentConfig(BaseModel):
    max_steps: int = 5
    timeout_seconds: int = 30
    memory_size: int = 10

class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    graph_db: GraphDBConfig = Field(default_factory=GraphDBConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    
    @classmethod
    def load_config(cls, config_path: str = "config/config.yaml") -> "Config":
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        return cls()