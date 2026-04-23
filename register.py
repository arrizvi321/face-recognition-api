import os
import json

# This builds enrolled image list.
# Goes through registered folder and builds JSON file of their labels and paths.

REGISTERED_DIR = "registered"
OUT_FILE = "outputs/registered_index.json"

def build_index():
    data = []
    for label in os.listdir(REGISTERED_DIR):
        label_path = os.path.join(REGISTERED_DIR, label)
        if not os.path.isdir(label_path):
            continue

        for fname in os.listdir(label_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                data.append({
                    "label": label,
                    "path": os.path.join(label_path, fname)
                })

    os.makedirs("outputs", exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} registered images to {OUT_FILE}")

if __name__ == "__main__":
    build_index()