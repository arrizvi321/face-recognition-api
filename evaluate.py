import json
import os
import time
import pandas as pd
from deepface import DeepFace

INDEX_FILE = "outputs/registered_index.json"
QUERY_IMAGES = [
    ("query/metest1.jpg", "me"),
    ("query/aloktest2.jpg", "alok")
]
MODELS = ["SFace", "ArcFace", "Facenet512"]
DETECTOR = "opencv" #opencv

def evaluate():

    with open(INDEX_FILE,"r",encoding="utf-8") as f:
        registered = json.load(f)

    rows = []

    for model in MODELS:

        for query_path, true_label in QUERY_IMAGES:

            closest_me = None
            closest_alok = None

            overall_best = None

            start = time.perf_counter()

            for item in registered:

                print("about to verify")
                result = DeepFace.verify(
                    img1_path=query_path,
                    img2_path=item["path"],
                    model_name=model,
                    detector_backend=DETECTOR,
                    enforce_detection=False
                )

                print("verified")
                dist = result["distance"]

                # overall closest image
                if overall_best is None or dist < overall_best["distance"]:
                    overall_best = {
                        "label": item["label"],
                        "path": item["path"],
                        "distance": dist
                    }

                # closest me
                if item["label"] == "me":
                    if closest_me is None or dist < closest_me:
                        closest_me = dist

                # closest other
                if item["label"] == "alok":
                    if closest_alok is None or dist < closest_alok:
                        closest_alok = dist

            elapsed_ms = (time.perf_counter() - start) * 1000

            predicted = overall_best["label"]
            correct = (predicted == true_label)

            separation_gap = None
            if closest_me is not None and closest_alok is not None:
                separation_gap = abs(closest_alok - closest_me)

            rows.append({
                "Model": model,
                "Detector": DETECTOR,

                "Query Image": query_path,

                "True Label": true_label,
                "Predicted Label": predicted,

                "Closest Me Distance": closest_me,
                "Closest Alok Distance": closest_alok,

                "Separation Gap": separation_gap,

                "Best Match Image": overall_best["path"],
                "Best Overall Distance": overall_best["distance"],

                "Correct": correct,

                "Total Compare Time ms": round(elapsed_ms,2),

                "Tested On": "Windows laptop CPU"
            })


    df = pd.DataFrame(rows)

    os.makedirs("outputs",exist_ok=True)

    out_file = "outputs/model_comparison.xlsx"

    df.to_excel(out_file,index=False)

    print(df)

    print(f"\nSaved {out_file}")

if __name__ == "__main__":
    evaluate()