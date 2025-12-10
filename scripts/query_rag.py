"""
RAG Query Script - Ask questions about your textbooks
Uses ChromaDB vectorstore + Local Llama model via Ollama
"""

import os
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import json

# Configuration
VECTORSTORE_PATH = Path("vectorstore")
COLLECTION_NAME = "textbook_embeddings"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LOCAL_MODEL_PATH = Path("models/bge-small-en-v1.5")
OLLAMA_MODEL = "llama-stu:latest"  # Change to your Ollama model name
OLLAMA_API_URL = "http://localhost:11434/api/generate" 
TOP_K_RESULTS = 5  # Number of relevant chunks to retrieve


class RAGQuerySystem:
    """RAG system for querying textbook knowledge."""
    
    def __init__(self):
        """Initialize the RAG system."""
        print("🚀 Initializing RAG Query System...")
        
        # Load embedding model
        print(f"  Loading embedding model: {EMBEDDING_MODEL}")
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"  ✓ Torch detected, using device: {device}")
        except:
            device = 'cpu'

        # try:
        #     device = 'cpu'
        # except:
        #     pass
        
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            cache_folder=str(LOCAL_MODEL_PATH.parent),
            device=device
        )
        print(f"  ✓ Embedding model loaded (device: {device})")
        
        # Connect to ChromaDB
        print(f"  Connecting to vectorstore: {VECTORSTORE_PATH}")
        self.client = chromadb.PersistentClient(
            path=str(VECTORSTORE_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        print(f"  ✓ Connected to collection: {COLLECTION_NAME}")
        print(f"  ✓ Total documents: {self.collection.count()}")
        print("\n✓ RAG System Ready!\n")
    
    def retrieve_context(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """Retrieve relevant context from vectorstore."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()
        
        # Search vectorstore
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        contexts = []
        for i in range(len(results['documents'][0])):
            contexts.append({
                'text': results['documents'][0][i],
                'source': results['metadatas'][0][i].get('source_file', 'Unknown'),
                'page': results['metadatas'][0][i].get('page', 'N/A'),
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return contexts
    
    def build_prompt(self, query: str, contexts: List[Dict]) -> str:
        """Build a prompt with retrieved context."""
        # Build context string
        context_str = "\n\n".join([
            f"[Source: {ctx['source']}, Page: {ctx['page']}]\n{ctx['text']}"
            for ctx in contexts
        ])
        
        # Simplified prompt - relies on Modelfile system prompt for personality
        prompt = f"""Relevant textbook content:
{context_str}

Question: {query}"""
        
        return prompt
    
    def query_ollama(self, prompt: str) -> str:
        """Send prompt to local Llama model via Ollama."""
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 1000
                }
            }
            
            response = requests.post(
                OLLAMA_API_URL,
                json=payload,
                timeout=120,
                stream=True
            )
            
            if response.status_code == 200:
                # result = response.json()
                # return result.get('response', 'No response from model')
                full_response = ""
                print("\n" +  "="*60)
                print("Answer:")
                print("="*60)

                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        text = chunk.get('response', '')
                        full_response += text
                        print(text, end='', flush=True)

                print("\n" +  "="*60 + "\n")
                return full_response

            else:
                return f"Error: Ollama API returned status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Make sure Ollama is running (run 'ollama serve')"
        except Exception as e:
            return f"Error querying Ollama: {str(e)}"
        
    #
    # """Send prompt to local Llama model via Ollama."""
    # try:
    #     payload = {
    #         "model": OLLAMA_MODEL,
    #         "prompt": prompt,
    #         "stream": False,
    #         "options": {
    #             "temperature": 0.7,
    #             "top_p": 0.9,
    #             "num_predict": 1000
    #         }
    #     }

    #     print(f"\n:mag: DEBUG - Sending to Ollama:")
    #     print(f"  Model: {OLLAMA_MODEL}")
    #     print(f"  URL: {OLLAMA_API_URL}")
    #     print(f"  Payload: {json.dumps(payload, indent=2)}")

    #     response = requests.post(
    #         OLLAMA_API_URL,
    #         json=payload,
    #         timeout=120
    #     )

    #     print(f"\n:satellite_antenna: DEBUG - Response:")
    #     print(f"  Status Code: {response.status_code}")
    #     print(f"  Response Text: {response.text[:500]}")  # First 500 chars

    #     if response.status_code == 200:
    #         result = response.json()
    #         return result.get('response', 'No response from model')
    #     else:
    #         return f"Error: Ollama API returned status {response.status_code}: {response.text}"

    # except requests.exceptions.ConnectionError:
    #     return "Error: Could not connect to Ollama. Make sure Ollama is running"
    # except Exception as e:
    #     return f"Error querying Ollama: {str(e)}"
    #
    
    def ask(self, question: str, show_sources: bool = True) -> Dict:
        """Main query function - retrieve context and generate answer."""
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}\n")
        
        # Step 1: Retrieve relevant context
        print(f"🔍 Retrieving relevant context (top {TOP_K_RESULTS} chunks)...")
        contexts = self.retrieve_context(question, top_k=TOP_K_RESULTS)
        print(f"✓ Retrieved {len(contexts)} relevant chunks\n")
        
        if show_sources:
            print("📚 Sources found:")
            for i, ctx in enumerate(contexts, 1):
                print(f"  {i}. {ctx['source']} (Page {ctx['page']})")
            print()
        
        # Step 2: Build prompt
        prompt = self.build_prompt(question, contexts)
        
        # Step 3: Query Ollama
        print(f"🤖 Asking {OLLAMA_MODEL}...")
        answer = self.query_ollama(prompt)
        
        print(f"\n{'='*60}")
        print("Answer:")
        print(f"{'='*60}")
        print(answer)
        print(f"{'='*60}\n")
        
        return {
            'question': question,
            'answer': answer,
            'contexts': contexts
        }


def interactive_mode():
    """Run interactive query session."""
    rag = RAGQuerySystem()
    
    print("=" * 60)
    print("Interactive RAG Query Mode")
    print("=" * 60)
    print("Ask questions about your Class 11 textbooks!")
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            question = input("Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Ask question
            result = rag.ask(question, show_sources=True)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


def single_query_mode(question: str):
    """Run a single query."""
    rag = RAGQuerySystem()
    result = rag.ask(question, show_sources=True)
    return result


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1:
        # Single query mode
        question = " ".join(sys.argv[1:])
        single_query_mode(question)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()