import json
import os

def scrape_shl_catalog():
    # Simulated scraper for SHL Individual Test Solutions
    raw_data = [
        {"title": "Verify Numerical Reasoning", "category": "Cognitive", "desc": "Measures the ability to make correct decisions or inferences from numerical or statistical data.", "time_limit": "25 mins"},
        {"title": "Verify Verbal Reasoning", "category": "Cognitive", "desc": "Measures the ability to evaluate the logic of various kinds of arguments.", "time_limit": "30 mins"},
        {"title": "OPQ32 (Occupational Personality Questionnaire)", "category": "Personality", "desc": "Provides a clear, simple framework for understanding the impact of personality on job performance.", "time_limit": "Untimed"},
        {"title": "Verify Inductive Reasoning", "category": "Cognitive", "desc": "Measures the ability to draw inferences and understand the relationships between various concepts.", "time_limit": "25 mins"},
        {"title": "Verify Deductive Reasoning", "category": "Cognitive", "desc": "Measures the ability to draw logical conclusions based on information provided.", "time_limit": "20 mins"},
        {"title": "Coding Simulator", "category": "Skills", "desc": "Evaluates candidate's coding skills in multiple programming languages.", "time_limit": "60 mins"},
        {"title": "Contact Center Simulation", "category": "Behavioral", "desc": "Assesses an individual's ability to handle customer service scenarios effectively.", "time_limit": "45 mins"}
    ]
    
    os.makedirs("data", exist_ok=True)
    with open("data/raw_catalog.json", "w") as f:
        json.dump(raw_data, f, indent=4)
        
    print("Scraped SHL catalog and saved to data/raw_catalog.json")

if __name__ == "__main__":
    scrape_shl_catalog()
