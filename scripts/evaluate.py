import sys
import os

# Ensure we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.retriever import retriever_service

SCENARIOS = [
    {
        "query": "We need to hire a software engineer and want to test their programming abilities.",
        "expected_target": "Coding Simulator"
    },
    {
        "query": "Looking for an assessment for a customer support role where they have to handle angry clients.",
        "expected_target": "Contact Center Simulation"
    },
    {
        "query": "I want to measure a candidate's personality traits to see if they fit our company culture.",
        "expected_target": "OPQ32 (Occupational Personality Questionnaire)"
    },
    {
        "query": "We need a test to see if candidates can understand and draw logical conclusions from statistical data.",
        "expected_target": "Verify Numerical Reasoning"
    }
]

def calculate_recall_at_k(scenarios, k=3):
    hits = 0
    total = len(scenarios)
    
    for scenario in scenarios:
        query = scenario["query"]
        expected = scenario["expected_target"]
        
        results = retriever_service.semantic_search(query, top_k=k)
        retrieved_names = [r["name"] for r in results]
        
        if expected in retrieved_names:
            hits += 1
            print(f"✅ HIT (Recall@{k}): '{query}' -> Found '{expected}'")
        else:
            print(f"❌ MISS (Recall@{k}): '{query}' -> Expected '{expected}', Got: {retrieved_names}")
            
    recall = hits / total if total > 0 else 0.0
    print(f"\nOverall Recall@{k}: {recall:.2f}")
    return recall

if __name__ == "__main__":
    print("Evaluating current retriever performance...")
    calculate_recall_at_k(SCENARIOS, k=1)
    calculate_recall_at_k(SCENARIOS, k=3)
