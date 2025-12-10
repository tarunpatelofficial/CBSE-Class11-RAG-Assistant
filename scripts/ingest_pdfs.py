"""
PDF Ingestion Script for RAG System
Processes PDFs and stores embeddings in ChromaDB for retrieval
"""

import os
from pathlib import Path
from typing import List
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuration
PDF_FOLDER = Path("Textbooks/Class_11")
VECTORSTORE_PATH = Path("vectorstore")
COLLECTION_NAME = "textbook_embeddings"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # BGE embeddings (will download locally)
LOCAL_MODEL_PATH = Path("models/bge-small-en-v1.5")  # Local cache for model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_pdfs(folder_path: Path) -> List:
    """Load all PDF files from the specified folder."""
    print(f"\n📂 Loading PDFs from {folder_path}...")
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    pdf_files = list(folder_path.glob("*.pdf"))
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {folder_path}")
    
    print(f"Found {len(pdf_files)} PDF files")
    
    all_documents = []
    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_path"] = str(pdf_file)
            
            all_documents.extend(documents)
            print(f"  ✓ Loaded {pdf_file.name}: {len(documents)} pages")
            
        except Exception as e:
            print(f"  ✗ Error loading {pdf_file.name}: {str(e)}")
    
    print(f"\n✓ Total pages loaded: {len(all_documents)}")
    return all_documents


def split_documents(documents: List, chunk_size: int, chunk_overlap: int) -> List:
    """Split documents into smaller chunks."""
    print(f"\n✂️  Splitting documents into chunks...")
    print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} chunks")
    
    return chunks


def load_local_embedding_model(model_name: str, cache_dir: Path):
    """Load embedding model locally (downloads once and caches)."""
    print(f"Loading embedding model: {model_name}")
    
    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model - will download on first run and cache locally
    model = SentenceTransformer(
        model_name,
        cache_folder=str(cache_dir.parent),
        device='cuda'  # Change to 'cpu' if no GPU available
    )
    
    print(f"✓ Model loaded (cached at: {cache_dir})")
    return model


def create_vectorstore(chunks: List, embedding_model: str, vectorstore_path: Path, collection_name: str):
    """Create ChromaDB vectorstore with embeddings."""
    print(f"\n🧠 Generating embeddings using {embedding_model}...")
    
    # Load local embedding model
    model = load_local_embedding_model(embedding_model, LOCAL_MODEL_PATH)
    
    # Create vectorstore directory if it doesn't exist
    vectorstore_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path=str(vectorstore_path),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(name=collection_name)
        print(f"  Deleted existing collection: {collection_name}")
    except:
        pass
    
    
    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"  Processing {len(chunks)} chunks...")
    
    # Process in batches to avoid memory issues
    batch_size = 100
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
        batch = chunks[i:i + batch_size]
        
        # Prepare texts and metadata
        texts = [chunk.page_content for chunk in batch]
        metadatas = [chunk.metadata for chunk in batch]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        
        # Generate embeddings using local model
        batch_embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()
        
        # Add to collection
        collection.add(
            embeddings=batch_embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    print(f"\n✓ Vectorstore created successfully!")
    print(f"  Location: {vectorstore_path}")
    print(f"  Collection: {collection_name}")
    print(f"  Total vectors: {collection.count()}")


def main():
    """Main ingestion pipeline."""
    print("=" * 60)
    print("PDF Ingestion Script for RAG System")
    print("=" * 60)
    
    try:
        # Step 1: Load PDFs
        documents = load_pdfs(PDF_FOLDER)
        
        # Step 2: Split into chunks
        chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # Step 3: Create vectorstore with embeddings
        create_vectorstore(chunks, EMBEDDING_MODEL, VECTORSTORE_PATH, COLLECTION_NAME)
        
        print("\n" + "=" * 60)
        print("✓ Ingestion completed successfully!")
        print("=" * 60)
        print(f"\nYou can now use the vectorstore for RAG queries.")
        print(f"Vectorstore location: {VECTORSTORE_PATH}")
        
    except Exception as e:
        print(f"\n❌ Error during ingestion: {str(e)}")
        raise


if __name__ == "__main__":
    main()