import json
import os
from typing import List, Dict

class CatalogLoader:
    def __init__(self, file_path: str = "data/structured_catalog.json"):
        self.file_path = file_path
        self.catalog = self._load_catalog()

    def _load_catalog(self) -> List[Dict]:
        if not os.path.exists(self.file_path):
            return []
        with open(self.file_path, "r") as f:
            return json.load(f)

    def get_all(self) -> List[Dict]:
        return self.catalog

    def search(self, query: str) -> List[Dict]:
        """Simple keyword search across name, description, and category."""
        query = query.lower()
        return [
            item for item in self.catalog
            if query in item["name"].lower() 
            or query in item["description"].lower()
            or query in item["category"].lower()
        ]

# Instantiate a global catalog loader to be used across the app
catalog_service = CatalogLoader()
