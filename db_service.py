import json
import os

DB_PATH = "database/face_db.json"

def load_db():
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) == 0:
        return []

    with open(DB_PATH, "r") as file:
        return json.load(file)

def save_db(db):
    with open(DB_PATH, "w") as file:
        json.dump(db, file, indent=4)