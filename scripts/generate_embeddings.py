import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def generate_embeddings():
    print("Loading catalog...")
    with open("data/structured_catalog.json", "r") as f:
        catalog = json.load(f)
        
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    texts = []
    for item in catalog:
        # Combine name, category, and description for richer embeddings
        text = f"{item['name']}. Category: {item['category']}. {item['description']}"
        texts.append(text)
        
    print(f"Generating embeddings for {len(texts)} items...")
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    print("Adding embeddings to FAISS index...")
    index.add(embeddings)
    
    faiss.write_index(index, "data/catalog.index")
    print("FAISS index saved to data/catalog.index")

if __name__ == "__main__":
    generate_embeddings()
