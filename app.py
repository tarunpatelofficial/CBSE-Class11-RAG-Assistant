"""
Student Helper RAG Application
A clean interface for querying textbooks using RAG
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import json
from datetime import datetime

# Configuration
VECTORSTORE_PATH = Path("vectorstore")
COLLECTION_NAME = "textbook_embeddings"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LOCAL_MODEL_PATH = Path("models/bge-small-en-v1.5")
OLLAMA_MODEL = "llama-stu:latest"  # Change to your preferred model
OLLAMA_API_URL = "http://localhost:11434/api/generate"
TOP_K_RESULTS = 3


class StudentHelperRAG:
    """Main RAG application for student textbook queries."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the RAG system.
        
        Args:
            verbose: Whether to print initialization messages
        """
        self.verbose = verbose
        self.is_initialized = False
        
        if self.verbose:
            print("\n" + "="*70)
            print("🎓 Student Helper - RAG System")
            print("="*70)
            print("Initializing...")
        
        try:
            # Load embedding model
            self._load_embedding_model()
            
            # Connect to vectorstore
            self._connect_vectorstore()
            
            # Verify Ollama connection
            self._verify_ollama()
            
            self.is_initialized = True
            
            if self.verbose:
                print("\n✅ System Ready!")
                print(f"📚 Loaded {self.collection.count()} textbook chunks")
                print(f"🤖 Using model: {OLLAMA_MODEL}")
                print("="*70 + "\n")
                
        except Exception as e:
            print(f"\n❌ Initialization Error: {str(e)}")
            raise
    
    def _load_embedding_model(self):
        """Load the sentence transformer model."""
        if self.verbose:
            print(f"  📥 Loading embedding model: {EMBEDDING_MODEL}")
        
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except:
            device = 'cpu'
        
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            cache_folder=str(LOCAL_MODEL_PATH.parent),
            device=device
        )
        
        if self.verbose:
            print(f"  ✓ Embedding model loaded (device: {device})")
    
    def _connect_vectorstore(self):
        """Connect to ChromaDB vectorstore."""
        if self.verbose:
            print(f"  📂 Connecting to vectorstore: {VECTORSTORE_PATH}")
        
        if not VECTORSTORE_PATH.exists():
            raise FileNotFoundError(
                f"Vectorstore not found at {VECTORSTORE_PATH}. "
                "Please run ingest_pdfs.py first."
            )
        
        self.client = chromadb.PersistentClient(
            path=str(VECTORSTORE_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        
        if self.verbose:
            print(f"  ✓ Connected to collection: {COLLECTION_NAME}")
    
    def _verify_ollama(self):
        """Verify Ollama is running and model is available."""
        if self.verbose:
            print(f"  🔌 Checking Ollama connection...")
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if OLLAMA_MODEL not in model_names:
                    print(f"\n  ⚠️  Warning: Model '{OLLAMA_MODEL}' not found!")
                    print(f"  Available models: {', '.join(model_names)}")
                    print(f"  Please update OLLAMA_MODEL in app.py or run:")
                    print(f"    ollama pull {OLLAMA_MODEL}")
                else:
                    if self.verbose:
                        print(f"  ✓ Ollama connected, model '{OLLAMA_MODEL}' available")
            else:
                raise Exception(f"Ollama returned status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            raise Exception(
                "Could not connect to Ollama. Please ensure Ollama is running.\n"
                "On Windows, Ollama should start automatically. If not, run 'ollama serve'"
            )
    
    def retrieve_context(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """
        Retrieve relevant context from vectorstore.
        
        Args:
            query: The search query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of context dictionaries with text, source, page, and distance
        """
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
                'distance': results['distances'][0][i] if 'distances' in results else None,
                'relevance_score': 1 - results['distances'][0][i] if 'distances' in results else None
            })
        
        return contexts
    
    def build_prompt(self, query: str, contexts: List[Dict]) -> str:
        """
        Build a prompt with retrieved context.
        
        Args:
            query: User's question
            contexts: List of retrieved context chunks
            
        Returns:
            Formatted prompt string
        """
        # Build context string
        context_str = "\n\n".join([
            f"[From {ctx['source']}, Page {ctx['page']}]\n{ctx['text']}"
            for ctx in contexts
        ])
        
        # Create prompt with strict academic focus
        prompt = f"""You are a Class 11 tutor. Answer ONLY based on the textbook content provided below.

TEXTBOOK CONTENT:
{context_str}

STUDENT'S QUESTION:
{query}

STRICT INSTRUCTIONS:
1. Answer ONLY if the textbook content above contains relevant information
2. If the content doesn't answer the question, say: "I couldn't find information about this in your Class 11 textbooks."
3. DO NOT answer questions about celebrities, sports players, current events, or topics not in textbooks
4. Stay focused on academic Class 11 subject matter
5. Start your answer directly without repeating the question

YOUR ANSWER:"""
        
        return prompt
    
    def generate_answer(self, prompt: str, stream: bool = False) -> str:
        """
        Generate answer using Ollama.
        
        Args:
            prompt: The formatted prompt
            stream: Whether to stream the response
            
        Returns:
            Generated answer text
        """
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 500,
                    "num_ctx": 2048
                }
            }
            
            if stream:
                return self._generate_streaming(payload)
            else:
                return self._generate_non_streaming(payload)
                
        except requests.exceptions.ConnectionError:
            return "❌ Error: Could not connect to Ollama. Please ensure Ollama is running."
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}"
    
    def _generate_non_streaming(self, payload: dict) -> str:
        """Generate answer without streaming."""
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', 'No response from model')
            # Clean up the response
            return self._clean_response(answer)
        else:
            error_msg = response.json().get('error', 'Unknown error')
            return f"❌ Ollama API Error ({response.status_code}): {error_msg}"
    
    def _clean_response(self, text: str) -> str:
        """Clean up model response."""
        # Remove common artifacts
        text = text.strip()
        
        # Remove incomplete sentence at start if it doesn't start with capital
        if text and not text[0].isupper() and not text[0].isdigit():
            # Find first sentence boundary
            first_period = text.find('. ')
            if first_period > 0 and first_period < 100:  # Only if within first 100 chars
                text = text[first_period + 2:].strip()
        
        return text
    
    def _generate_streaming(self, payload: dict) -> str:
        """Generate answer with streaming output."""
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=120,
            stream=True
        )
        
        if response.status_code == 200:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    text = chunk.get('response', '')
                    full_response += text
                    print(text, end='', flush=True)
            print()  # New line after streaming
            return full_response
        else:
            error_msg = response.json().get('error', 'Unknown error')
            return f"❌ Ollama API Error ({response.status_code}): {error_msg}"
    
    def ask(self, question: str, show_sources: bool = True, 
            show_context: bool = False, stream: bool = False) -> Dict:
        """
        Main query function.
        
        Args:
            question: User's question
            show_sources: Whether to display source information
            show_context: Whether to display retrieved context text
            stream: Whether to stream the answer
            
        Returns:
            Dictionary containing question, answer, contexts, and metadata
        """
        if not self.is_initialized:
            return {
                'error': 'System not initialized properly',
                'question': question
            }
        
        start_time = datetime.now()
        
        # Print query header
        print("\n" + "="*70)
        print(f"❓ Question: {question}")
        print("="*70)
        
        # Step 1: Retrieve relevant context
        print(f"\n🔍 Searching textbooks (retrieving top {TOP_K_RESULTS} matches)...")
        contexts = self.retrieve_context(question, top_k=TOP_K_RESULTS)
        print(f"✓ Found {len(contexts)} relevant sections")
        
        # Check relevance threshold
        if contexts and contexts[0]['relevance_score'] is not None:
            best_score = contexts[0]['relevance_score']
            if best_score < 0.65:  # Less than 65% relevance
                print(f"\n⚠️  Low relevance detected (best match: {best_score:.1%})")
                answer = (
                    "I couldn't find relevant information about this topic in your Class 11 textbooks. "
                    "This question might be:\n"
                    "- Outside the Class 11 syllabus\n"
                    "- About a topic not covered in the available textbooks\n"
                    "- Unrelated to academic subjects\n\n"
                    "Please ask questions related to your Class 11 subjects like Biology, "
                    "Accountancy, Computer Science, English, etc."
                )
                
                end_time = datetime.now()
                time_taken = (end_time - start_time).total_seconds()
                
                print("\n" + "-"*70)
                print("💬 Answer:")
                print("-"*70)
                print(answer)
                print("-"*70)
                print(f"\n⏱️  Time taken: {time_taken:.2f} seconds")
                print("="*70 + "\n")
                
                return {
                    'question': question,
                    'answer': answer,
                    'contexts': contexts,
                    'time_taken': time_taken,
                    'model': OLLAMA_MODEL,
                    'timestamp': start_time.isoformat(),
                    'low_relevance': True
                }
        
        # Show sources
        if show_sources and contexts:
            print("\n📚 Sources:")
            for i, ctx in enumerate(contexts, 1):
                relevance = f" (relevance: {ctx['relevance_score']:.2%})" if ctx['relevance_score'] else ""
                print(f"  {i}. {ctx['source']} - Page {ctx['page']}{relevance}")
        
        # Show context (optional, for debugging)
        if show_context and contexts:
            print("\n📖 Retrieved Context:")
            for i, ctx in enumerate(contexts, 1):
                print(f"\n  Context {i}:")
                print(f"  {ctx['text'][:200]}...")
        
        # Step 2: Build prompt
        prompt = self.build_prompt(question, contexts)
        
        # Step 3: Generate answer
        print(f"\n🤖 Generating answer using {OLLAMA_MODEL}...")
        if stream:
            print("\n" + "-"*70)
            print("💬 Answer:")
            print("-"*70)
        
        answer = self.generate_answer(prompt, stream=stream)
        
        if not stream:
            print("\n" + "-"*70)
            print("💬 Answer:")
            print("-"*70)
            print(answer)
            print("-"*70)
        
        # Calculate time taken
        end_time = datetime.now()
        time_taken = (end_time - start_time).total_seconds()
        
        print(f"\n⏱️  Time taken: {time_taken:.2f} seconds")
        print("="*70 + "\n")
        
        return {
            'question': question,
            'answer': answer,
            'contexts': contexts,
            'time_taken': time_taken,
            'model': OLLAMA_MODEL,
            'timestamp': start_time.isoformat()
        }
    
    def batch_ask(self, questions: List[str]) -> List[Dict]:
        """
        Ask multiple questions in batch.
        
        Args:
            questions: List of questions
            
        Returns:
            List of result dictionaries
        """
        results = []
        total = len(questions)
        
        print(f"\n📝 Processing {total} questions...")
        
        for i, question in enumerate(questions, 1):
            print(f"\n[Question {i}/{total}]")
            result = self.ask(question, show_sources=True, stream=False)
            results.append(result)
        
        return results


def interactive_mode():
    """Run interactive query session."""
    rag = StudentHelperRAG(verbose=True)
    
    print("\n💡 Tips:")
    print("  - Ask questions about your Class 11 textbooks")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'sources on/off' to toggle source display")
    print("  - Type 'stream on/off' to toggle streaming")
    print()
    
    show_sources = True
    stream_mode = False
    
    while True:
        try:
            user_input = input("🎓 Your question: ").strip()
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using Student Helper! Good luck with your studies!")
                break
            
            if user_input.lower() == 'sources on':
                show_sources = True
                print("✓ Source display enabled\n")
                continue
            
            if user_input.lower() == 'sources off':
                show_sources = False
                print("✓ Source display disabled\n")
                continue
            
            if user_input.lower() == 'stream on':
                stream_mode = True
                print("✓ Streaming mode enabled\n")
                continue
            
            if user_input.lower() == 'stream off':
                stream_mode = False
                print("✓ Streaming mode disabled\n")
                continue
            
            if not user_input:
                continue
            
            # Ask question
            rag.ask(user_input, show_sources=show_sources, stream=stream_mode)
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using Student Helper!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


def single_query_mode(question: str):
    """Run a single query."""
    rag = StudentHelperRAG(verbose=True)
    result = rag.ask(question, show_sources=True, stream=False)
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
