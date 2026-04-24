import json
from pathlib import Path
from typing import List, Dict, Any

from deepface import DeepFace

# -----------------------------
# Config
# -----------------------------
REGISTERED_DIR = "Images/registered"
OUTPUT_JSON = "database/face_db.json"

MODEL_NAME = "SFace"
DETECTOR_BACKEND = "opencv" 
#DISTANCE_METRIC = "cosine"
ENFORCE_DETECTION = True



def get_embedding(image_path: str) -> List[float]:
    """
    Generate one embedding vector from an image using DeepFace.
    """
    result = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=ENFORCE_DETECTION
    )

    if not result or "embedding" not in result[0]:
        raise ValueError(f"No embedding returned for {image_path}")

    return result[0]["embedding"]


def build_database(registered_dir: str) -> List[Dict[str, Any]]:
    """
    Creates one JSON entry per image embedding.
    """
    db: List[Dict[str, Any]] = []
    root = Path(registered_dir)

    if not root.exists():
        raise FileNotFoundError(f"Registered folder not found: {registered_dir}")

    people_dirs = [p for p in root.iterdir() if p.is_dir()]
    people_dirs.sort()

    for person_dir in people_dirs:
        person_name = person_dir.name
        image_files = [p for p in person_dir.rglob("*")]
        image_files.sort()

        if not image_files:
            print(f"[WARN] No images found for {person_name}")
            continue

        print(f"\nProcessing person: {person_name}")

        for img_path in image_files:
            try:
                embedding = get_embedding(str(img_path))

                db.append({
                    "name": person_name,
                    "image_path": str(img_path).replace("\\", "/"),
                    "model": MODEL_NAME,
                    "detector": DETECTOR_BACKEND,
                    #"distance_metric": DISTANCE_METRIC,
                    "embedding": embedding
                })

                print(f"  [OK] {img_path.name}")

            except Exception as e:
                print(f"  [FAIL] {img_path.name} -> {e}")

    return db


def save_database(db: List[Dict[str, Any]], output_json: str) -> None:
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
    print(f"\nSaved {len(db)} embeddings to {output_json}")


def main() -> None:
    db = build_database(REGISTERED_DIR)
    save_database(db, OUTPUT_JSON)


if __name__ == "__main__":
    main()