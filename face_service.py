import math
import os
import tempfile
from typing import List, Dict, Any, Optional

from fastapi import UploadFile
from deepface import DeepFace

from db_service import load_db, save_db

MODEL_NAME = "SFace"
DETECTOR_BACKEND = "opencv"
DISTANCE_METRIC = "cosine"
ENFORCE_DETECTION = True

THRESHOLD = 0.35

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def validate_filename(filename: Optional[str]) -> None:
    if not filename:
        return

    ext = os.path.splitext(filename)[1].lower()

    if ext and ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")


def save_upload_temporarily(upload: UploadFile) -> str:
    validate_filename(upload.filename)

    suffix = os.path.splitext(upload.filename or "image.jpg")[1] or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = upload.file.read()

        if not content:
            raise ValueError("Uploaded file is empty")

        tmp.write(content)
        return tmp.name


def cosine_distance(vec1: List[float], vec2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 1.0

    return 1.0 - (dot / (norm1 * norm2))


def get_embedding_from_image(image_path: str) -> List[float]:
    result = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=ENFORCE_DETECTION
    )

    if not result or "embedding" not in result[0]:
        raise ValueError("No embedding returned")

    return result[0]["embedding"]


def register_person(person_id: str, image: UploadFile) -> Dict[str, Any]:
    tmp_path = None

    try:
        if not person_id.strip():
            raise ValueError("person_id cannot be empty")

        db = load_db()

        tmp_path = save_upload_temporarily(image)
        embedding = get_embedding_from_image(tmp_path)

        record = {
            "person_id": person_id.strip(),
            "model": MODEL_NAME,
            "detector": DETECTOR_BACKEND,
            "distance_metric": DISTANCE_METRIC,
            "source_filename": image.filename,
            "embedding": embedding
        }

        db.append(record)
        save_db(db)

        vectors_for_person = sum(
            1 for entry in db if entry["person_id"] == person_id.strip()
        )

        return {
            "success": True,
            "message": "Face registered successfully",
            "person_id": person_id.strip(),
            "vectors_for_person": vectors_for_person,
            "total_vectors": len(db)
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def recognize_person(image: UploadFile) -> Dict[str, Any]:
    tmp_path = None

    try:
        db = load_db()

        if not db:
            raise ValueError("Database is empty")

        tmp_path = save_upload_temporarily(image)
        query_embedding = get_embedding_from_image(tmp_path)

        best_entry = None
        best_distance = float("inf")

        for entry in db:
            dist = cosine_distance(query_embedding, entry["embedding"])

            if dist < best_distance:
                best_distance = dist
                best_entry = entry

        if best_entry is None:
            return {
                "success": False,
                "message": "No match found"
            }

        is_match = best_distance <= THRESHOLD

        return {
            "success": True,
            "match": is_match,
            "person_id": best_entry["person_id"] if is_match else None,
            "best_candidate": best_entry["person_id"],
            "distance": best_distance,
            "threshold": THRESHOLD,
            "model": MODEL_NAME,
            "detector": DETECTOR_BACKEND
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def verify_person(person_id: str, image: UploadFile) -> Dict[str, Any]:
    tmp_path = None

    try:
        db = load_db()

        candidates = [
            entry for entry in db
            if entry.get("person_id") == person_id
        ]

        if not candidates:
            raise ValueError(f"No registered vectors found for '{person_id}'")

        tmp_path = save_upload_temporarily(image)
        query_embedding = get_embedding_from_image(tmp_path)

        best_distance = min(
            cosine_distance(query_embedding, entry["embedding"])
            for entry in candidates
        )

        verified = best_distance <= THRESHOLD

        return {
            "success": True,
            "verified": verified,
            "person_id": person_id,
            "distance": best_distance,
            "threshold": THRESHOLD,
            "model": MODEL_NAME,
            "detector": DETECTOR_BACKEND
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def list_persons() -> Dict[str, Any]:
    db = load_db()

    persons = {}

    for entry in db:
        person_id = entry.get("person_id", "unknown")
        persons[person_id] = persons.get(person_id, 0) + 1

    return {
        "success": True,
        "persons": persons,
        "total_vectors": len(db)
    }


def delete_person(person_id: str) -> Dict[str, Any]:
    db = load_db()

    new_db = [
        entry for entry in db
        if entry.get("person_id") != person_id
    ]

    removed_count = len(db) - len(new_db)

    if removed_count == 0:
        raise ValueError(f"No entries found for '{person_id}'")

    save_db(new_db)

    return {
        "success": True,
        "message": "Person deleted from database",
        "person_id": person_id,
        "removed_vectors": removed_count,
        "remaining_vectors": len(new_db)
    }