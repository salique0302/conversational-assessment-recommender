import json
import uuid

def preprocess_catalog():
    with open("data/raw_catalog.json", "r") as f:
        raw_data = json.load(f)
        
    structured_data = []
    for item in raw_data:
        # Create a deterministic UUID based on the item title
        item_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item["title"]))
        
        structured_item = {
            "id": item_id,
            "name": item["title"],
            "description": item["desc"],
            "category": item["category"],
            "url": item.get("url", ""),
            "test_type": item.get("test_type", item["category"]),
            "metadata": {
                "time_limit": item["time_limit"]
            }
        }
        structured_data.append(structured_item)
        
    with open("data/structured_catalog.json", "w") as f:
        json.dump(structured_data, f, indent=4)
        
    print("Preprocessed catalog and saved to data/structured_catalog.json")

if __name__ == "__main__":
    preprocess_catalog()
