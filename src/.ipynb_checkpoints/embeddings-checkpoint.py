# src/embeddings.py
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os
from pathlib import Path
from typing import List, Dict
import json

# Initialize the embedding model
# This model is free, runs locally, and works well for search
def get_embedding_model():
    """Load the sentence transformer model"""
    print("🔄 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded")
    return model

def create_chroma_db():
    """Create or connect to ChromaDB"""
    # Create db directory if it doesn't exist
    db_path = Path("./chroma_db")
    db_path.mkdir(exist_ok=True)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=str(db_path))
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name="montessori_docs",
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    
    print(f"✅ Connected to ChromaDB at {db_path}")
    return collection

def add_documents_to_db(collection, chunks: List[Dict], model):
    """Convert chunks to embeddings and add to database"""
    
    # Prepare data
    texts = [chunk['text'] for chunk in chunks]
    ids = [f"doc_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": chunk['source'],
            "title": chunk['title'],
            "chunk_id": chunk['chunk_id']
        }
        for chunk in chunks
    ]
    
    # Generate embeddings (this may take a minute)
    print(f"🔄 Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts).tolist()
    print("✅ Embeddings generated")
    
    # Add to database
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    
    print(f"✅ Added {len(chunks)} chunks to database")
    return collection

def search_similar(collection, query: str, model, n_results: int = 3):
    """Search for similar chunks"""
    # Generate query embedding
    query_embedding = model.encode(query).tolist()
    
    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return results

if __name__ == "__main__":
    # Test the embeddings pipeline
    print("🧪 Testing embeddings pipeline...")
    
    # Load some test data
    from data_loader import load_text_files, process_documents
    
    docs = load_text_files("data/raw")
    if not docs:
        print("❌ No documents found. Add some .txt files to data/raw/")
        exit()
    
    chunks = process_documents(docs)
    
    # Initialize model and DB
    model = get_embedding_model()
    collection = create_chroma_db()
    
    # Add documents
    add_documents_to_db(collection, chunks, model)
    
    # Test search
    test_query = "What are sensitive periods in Montessori?"
    print(f"\n🔍 Testing search with: '{test_query}'")
    results = search_similar(collection, test_query, model)
    
    # Show results
    if results['documents']:
        print("\n📝 Top result:")
        print(results['documents'][0][0][:300] + "...")
        print(f"\n📊 Metadata: {results['metadatas'][0][0]}")