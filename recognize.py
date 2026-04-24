import json
import math
from typing import List, Dict, Any, Tuple
from deepface import DeepFace

# -----------------------------
# Config
# -----------------------------
DB_JSON = "database/face_db.json"

MODEL_NAME = "SFace"
DETECTOR_BACKEND = "opencv"
ENFORCE_DETECTION = True

# Tune this threshold based on your test results
# For cosine distance, lower is better.
THRESHOLD = 0.35


def load_database(db_json: str) -> List[Dict[str, Any]]:
    with open(db_json, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_distance(vec1: List[float], vec2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 1.0

    cosine_similarity = dot / (norm1 * norm2)
    return 1.0 - cosine_similarity


def get_embedding(image_path: str) -> List[float]:
    result = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=ENFORCE_DETECTION
    )

    if not result or "embedding" not in result[0]:
        raise ValueError(f"No embedding returned for {image_path}")

    return result[0]["embedding"]


def find_best_match(
    query_embedding: List[float],
    db: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], float]:
    best_entry = None
    best_distance = float("inf")

    for entry in db:
        dist = cosine_distance(query_embedding, entry["embedding"])
        if dist < best_distance:
            best_distance = dist
            best_entry = entry

    return best_entry, best_distance


def recognize_image(image_path: str) -> None:
    db = load_database(DB_JSON)

    if not db:
        print("Database is empty.")
        return

    query_embedding = get_embedding(image_path)
    best_entry, best_distance = find_best_match(query_embedding, db)

    if best_entry is None:
        print("No match found.")
        return

    print("\nRecognition result")
    print("------------------")
    print(f"Test image      : {image_path}")
    print(f"Best match      : {best_entry['name']}")
    print(f"Matched image   : {best_entry['image_path']}")
    print(f"Distance        : {best_distance:.6f}")
    print(f"Threshold       : {THRESHOLD:.6f}")

    if best_distance <= THRESHOLD:
        print("Decision        : ACCEPTED")
    else:
        print("Decision        : UNKNOWN")


if __name__ == "__main__":
    test_image = "Images/test/johnydepptest1.jpg"   # change this
    recognize_image(test_image)