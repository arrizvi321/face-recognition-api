import json
from deepface import DeepFace

# This runs one model on one query image.

INDEX_FILE = "outputs/registered_index.json"
QUERY_IMG = "query/metest1.jpg"
MODEL_NAME = "SFace"  # ArcFace, SFace, Facenet512

print("Image to be tested is:", QUERY_IMG)

def recognize():
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        registered = json.load(f)

    best = None

    for item in registered:
        result = DeepFace.verify(
            img1_path=QUERY_IMG,
            img2_path=item["path"],
            model_name=MODEL_NAME,
            detector_backend="opencv",
            enforce_detection=False
        )

        row = {
            "label": item["label"],
            "path": item["path"],
            "distance": result["distance"],
            "verified": result["verified"]
        }

        if best is None or row["distance"] < best["distance"]:
            best = row

        print(row)

    print("\nBest match:")
    print(best)

if __name__ == "__main__":
    recognize()