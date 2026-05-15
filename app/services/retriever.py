import os
import json
import faiss
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder

class RetrieverService:
    def __init__(self, catalog_path: str = "data/structured_catalog.json", index_path: str = "data/catalog.index"):
        self.catalog_path = catalog_path
        self.index_path = index_path
        
        # Load catalog
        with open(self.catalog_path, "r") as f:
            self.catalog = json.load(f)
            
        # Load FAISS index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = None
            
        # Initialize embedding model for queries
        print("Loading SentenceTransformer model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize reranker model
        print("Loading CrossEncoder model...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def semantic_search(self, query: str, top_k: int = 5, filters: Optional[Dict[str, str]] = None) -> List[Dict]:
        if not self.index:
            return []
            
        # 1. Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # 2. FAISS retrieval (get top 10 to optimize CrossEncoder latency)
        retrieve_k = min(10, len(self.catalog))
        distances, indices = self.index.search(query_embedding, retrieve_k)
        
        retrieved_items = []
        for idx in indices[0]:
            if idx == -1:
                continue
            retrieved_items.append(self.catalog[idx])
            
        # 3. Apply metadata filtering
        if filters:
            filtered_items = []
            for item in retrieved_items:
                match = True
                for key, value in filters.items():
                    # Check direct attributes and metadata
                    item_val = item.get(key) or item.get("metadata", {}).get(key)
                    if item_val and str(item_val).lower() != str(value).lower():
                        match = False
                        break
                if match:
                    filtered_items.append(item)
            retrieved_items = filtered_items
            
        if not retrieved_items:
            return []
            
        # 4. Reranking using CrossEncoder
        cross_inp = [[query, f"{item['name']} - {item['description']}"] for item in retrieved_items]
        scores = self.reranker.predict(cross_inp)
        
        # Combine items with scores and apply a lexical overlap boost
        scored_items = []
        for item, score in zip(retrieved_items, scores):
            query_lower = query.lower()
            name_lower = item['name'].lower()
            
            # Simple lexical boost for keyword matches
            overlap_bonus = 0.0
            for term in query_lower.split():
                if len(term) > 3 and term in name_lower:
                    overlap_bonus += 2.0
                    
            # Domain specific synonym handling
            if "statistical" in query_lower and "numerical" in name_lower:
                overlap_bonus += 3.0
                
            final_score = float(score) + overlap_bonus
            scored_items.append((item, final_score))
            
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        return [item for item, score in scored_items[:top_k]]

# Instantiate a global retriever service to be used across the app
try:
    retriever_service = RetrieverService()
except Exception as e:
    print(f"Warning: Could not initialize RetrieverService: {e}")
    retriever_service = None
