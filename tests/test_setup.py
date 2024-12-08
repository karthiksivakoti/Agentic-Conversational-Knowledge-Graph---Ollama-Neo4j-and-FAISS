import sys
import torch
from neo4j import GraphDatabase
from transformers import AutoTokenizer
import requests  # We'll use requests instead of ollama package
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cuda():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

def test_neo4j():
    try:
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password123")
        )
        with driver.session() as session:
            result = session.run("RETURN 1")
            print("Neo4j connection successful")
    except Exception as e:
        print("Neo4j connection failed:", e)
    finally:
        if 'driver' in locals():
            driver.close()

def test_ollama():
    """Test Ollama connection using REST API"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            print("Ollama connection successful")
            print("Available models:", models)
        else:
            print("Ollama connection failed with status code:", response.status_code)
    except requests.exceptions.ConnectionError:
        print("Ollama connection failed: Is Ollama running?")
    except Exception as e:
        print("Ollama test failed:", e)

def test_transformers():
    try:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        print("Transformers setup successful")
    except Exception as e:
        print("Transformers setup failed:", e)

if __name__ == "__main__":
    print("\n=== Environment Setup Test ===")
    print("Python version:", sys.version)
    print("\n1. Testing CUDA Setup:")
    test_cuda()
    print("\n2. Testing Neo4j Connection:")
    test_neo4j()
    print("\n3. Testing Ollama Connection:")
    test_ollama()
    print("\n4. Testing Transformers Setup:")
    test_transformers()