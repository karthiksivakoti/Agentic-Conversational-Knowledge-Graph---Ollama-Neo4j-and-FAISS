#src/ingestion/loaders/document_loaders.py
from typing import List, Dict, Any
import os
from pathlib import Path
import logging
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
import json

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Loads documents from various file formats"""

    supported_extensions = {'.pdf', '.docx', '.txt', '.json', '.md'}

    @staticmethod
    async def load_directory(directory_path: str) -> List[Dict[str, Any]]:
        """Load all supported documents from a directory"""
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in DocumentLoader.supported_extensions:
                try:
                    content = await DocumentLoader._read_file(file_path)
                    if content:
                        documents.append({
                            'content': content,
                            'metadata': {
                                'source': str(file_path),
                                'file_type': file_path.suffix[1:],  # Remove the dot
                                'file_name': file_path.name
                            }
                        })
                        logger.info(f"Successfully loaded document: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}")

        return documents

    @staticmethod
    async def _read_file(file_path: Path) -> str:
        """Read content from file based on its type"""
        try:
            if file_path.suffix.lower() == '.pdf':
                reader = PdfReader(file_path)
                return ' '.join(page.extract_text() for page in reader.pages)

            elif file_path.suffix.lower() == '.docx':
                doc = DocxDocument(file_path)
                return ' '.join(paragraph.text for paragraph in doc.paragraphs)

            elif file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()

            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Assume the JSON has a 'content' field, adjust as needed
                    return data.get('content', '')

            elif file_path.suffix.lower() == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise