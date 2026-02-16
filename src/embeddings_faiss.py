# src/embeddings_faiss.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from pathlib import Path
from typing import List, Dict
import json

# Initialize the embedding model
def get_embedding_model():
    """Load the sentence transformer model"""
    print("🔄 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded")
    return model

class FAISSDatabase:
    def __init__(self, model, index_path="./faiss_index"):
        self.model = model
        self.index_path = Path(index_path)
        self.index_path.mkdir(exist_ok=True)
        
        self.index_file = self.index_path / "index.faiss"
        self.metadata_file = self.index_path / "metadata.pkl"
        
        # Initialize or load index
        if self.index_file.exists() and self.metadata_file.exists():
            self.load()
        else:
            self.index = None
            self.documents = []
            self.metadata = []
    
    def add_documents(self, chunks: List[Dict]):
        """Add documents to FAISS index"""
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        print(f"🔄 Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts)
        print(f"✅ Generated embeddings with shape {embeddings.shape}")
        
        # Initialize index if first time
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
            print(f"✅ Created new FAISS index with dimension {dimension}")
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(texts)
        self.metadata.extend([
            {
                "source": chunk['source'],
                "title": chunk['title'],
                "chunk_id": chunk['chunk_id']
            }
            for chunk in chunks
        ])
        
        print(f"✅ Added {len(chunks)} chunks to index")
        self.save()
    
    def search(self, query: str, k: int = 3):
        """Search for similar documents"""
        if self.index is None or self.index.ntotal == 0:
            print("⚠️ Index is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                results.append({
                    "text": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "score": float(distances[0][i])
                })
        
        return results
    
    def save(self):
        """Save index and metadata"""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_file))
            with open(self.metadata_file, 'wb') as f:
                pickle.dump((self.documents, self.metadata), f)
            print(f"✅ Saved index to {self.index_path}")
    
    def load(self):
        """Load index and metadata"""
        if self.index_file.exists() and self.metadata_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            with open(self.metadata_file, 'rb') as f:
                self.documents, self.metadata = pickle.load(f)
            print(f"✅ Loaded index with {self.index.ntotal} documents from {self.index_path}")
        else:
            print(f"⚠️ No existing index found at {self.index_path}")

def main():
    """Test the FAISS pipeline"""
    print("🧪 Testing FAISS embeddings pipeline...")
    
    # Load test data
    from data_loader import load_text_files, process_documents
    
    docs = load_text_files("data/raw")
    if not docs:
        print("❌ No documents found. Add some .txt files to data/raw/")
        return
    
    chunks = process_documents(docs)
    
    # Initialize model and database
    model = get_embedding_model()
    db = FAISSDatabase(model)
    
    # Add documents
    db.add_documents(chunks)
    
    # Test search
    test_queries = [
        "What are sensitive periods?",
        "How do I handle a tantrum?",
        "What is the absorbent mind?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 50)
        
        results = db.search(query, k=2)
        
        for i, result in enumerate(results):
            print(f"\nResult {i+1} (from {result['metadata']['title']}):")
            print(f"Score: {result['score']:.4f}")
            print(result['text'][:200] + "...")
        
        print("\n" + "=" * 50)

if __name__ == "__main__":
    main()