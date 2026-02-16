import os
from pathlib import Path
from typing import List, Dict

def load_text_files(data_path: str = "data/raw") -> List[Dict]:
    """Load all text files from the raw data folder"""
    documents = []
    data_dir = Path(data_path)
    
    if not data_dir.exists():
        print(f"Warning: {data_path} directory not found!")
        return documents
    
    for file_path in data_dir.glob("*.txt"):
        print(f"Loading {file_path.name}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                if text.strip():  # Only add if not empty
                    documents.append({
                        "text": text,
                        "source": file_path.name,
                        "title": file_path.stem.replace('_', ' ').title()
                    })
                    print(f"  ✅ Loaded {len(text)} characters")
                else:
                    print(f"  ⚠️ File is empty")
        except Exception as e:
            print(f"  ❌ Error loading {file_path.name}: {e}")
    
    print(f"\n✅ Loaded {len(documents)} documents total")
    return documents

def simple_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Only keep chunks with meaningful content
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        
        start += (chunk_size - overlap)
    
    return chunks

def process_documents(documents: List[Dict]) -> List[Dict]:
    """Process all documents into chunks"""
    all_chunks = []
    
    for doc in documents:
        chunks = simple_chunk(doc["text"])
        print(f"  📄 {doc['title']}: {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": doc["source"],
                "title": doc["title"],
                "chunk_id": i
            })
    
    print(f"\n✅ Created {len(all_chunks)} total chunks")
    return all_chunks

if __name__ == "__main__":
    # Test the pipeline
    print("📚 Loading documents...")
    docs = load_text_files()
    
    if docs:
        print("\n🔪 Chunking documents...")
        chunks = process_documents(docs)
        
        # Show a sample
        if chunks:
            print("\n📝 Sample chunk:")
            print("-" * 40)
            print(f"Source: {chunks[0]['title']}")
            print(f"Text: {chunks[0]['text'][:200]}...")
    else:
        print("❌ No documents found. Add some .txt files to data/raw/")