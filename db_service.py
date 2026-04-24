import json
import os
from typing import List, Dict, Any

DB_PATH = "database/face_db.json"


def load_db() -> List[Dict[str, Any]]:
    if not os.path.exists(DB_PATH):
        return []

    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(data: List[Dict[str, Any]]) -> None:
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)