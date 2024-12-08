# Agentic RAG: Retrieval-Augmented Generation with Multi-Agent Orchestration 🤖
[![LangChain](https://img.shields.io/badge/LangChain-%23121011.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI2MDAiIGhlaWdodD0iNjAwIiB2aWV3Qm94PSIwIDAgNjAwIDYwMCI+CiAgPHBhdGggZmlsbD0iI2ZmZiIgZD0iTTMwMCA1MGMxMzcuNSAwIDI1MCAxMTIuNSAyNTAgMjUwcy0xMTIuNSAyNTAtMjUwIDI1MFM1MCAzODcuNSA1MCAyNTAgMTYyLjUgNTAgMzAwIDUweiIvPgo8L3N2Zz4K)](https://langchain.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-%234EA94B.svg?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-%23000000.svg?style=for-the-badge&logo=llama&logoColor=white)](https://ollama.ai/)
[![FAISS](https://img.shields.io/badge/FAISS-%23150458.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://github.com/facebookresearch/faiss)

## Overview 🎯
This repository implements an advanced Retrieval-Augmented Generation (RAG) system using a multi-agent architecture. The system combines vector search, knowledge graphs, and intelligent orchestration to provide enhanced document understanding and query processing capabilities.

## Key Features ⚡
- **Multi-Agent Orchestration**: Coordinated workflow between specialized agents (query understanding, research, test generation, validation) for modular and scalable solutions.
- **Hybrid Search**: Integrates vector store (FAISS) for semantic similarity search and Neo4j knowledge graph queries for rich, relational context.
- **Knowledge Graph Integration**: Leverages Neo4j to store and query entities, relationships, and contextual details extracted from documents.
- **Intelligent Document Processing**: Uses `unstructured` for parsing diverse file formats (PDF, DOCX, TXT, JSON, images, audio/video transcripts) and advanced entity extraction.
- **Source Attribution**: Responses include source citations from ingested documents, ensuring transparency and traceability.
- **Automated Testing & Validation**: Agents generate and validate test cases to ensure the reliability and accuracy of responses.

## Architecture 🏗️
1. **Ingestion Pipeline**:  
   - Parses raw documents from `data/raw` directory (PDFs, DOCX, images, audio/video transcripts).
   - Splits text into chunks, enriches with embeddings, entities, and stores in vector store & Neo4j graph.
   
2. **Agents**:  
   - **Query Agent**: Interprets user queries, identifies required knowledge components.  
   - **Research Agent**: Fetches relevant chunks from vector store & related entities from graph.  
   - **Test Agent**: Generates and validates test cases to ensure answer integrity.  
   - **Validator Agent**: Validates final responses for accuracy, logical consistency, and source attribution.
   
3. **Hybrid Search & RAG**:  
   - Combines vector-based semantic search with graph-based context retrieval.
   - Prepares a prompt for the LLM (Ollama or OpenAI) with relevant context and sources.
   - The LLM produces an answer citing sources from the loaded documents.

## Getting Started 🚀
### Prerequisites
- Python 3.12+
- Neo4j running locally (default: `bolt://localhost:7687`)
- FAISS for vector indexing
- Ollama for local model inference (or configure OpenAI keys for GPT-4)
- `unstructured` for document parsing
- `pytesseract` + `tesseract` binary for OCR (if processing images)
- `whisper` for audio/video transcription (optional)

### Installation
```bash
git clone https://github.com/karthiksivakoti/agentic_rag.git
cd agentic_rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### Configuration
Modify `config/config.yaml` or environment variables for:

- Neo4j credentials
- LLM settings (Ollama model or OpenAI API keys)
- Vector store location

### Running the Pipeline
```bash
pytest tests/integration/test_rag_pipeline.py -v --log-cli-level=INFO
```
This runs the full ingestion pipeline, multi-agent orchestration tests, and ensures everything works end-to-end.

## Usage 💻
Once ingested, you can:
- Run a query through the pipeline, get an LLM-generated answer with cited sources.
- Extend agents for more specialized tasks.
- Integrate into a UI (like Streamlit) for conversational retrieval.

## License 📜
This project is licensed under the MIT License
