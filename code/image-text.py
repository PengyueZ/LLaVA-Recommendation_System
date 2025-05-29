import os
import json
from datasets import load_dataset

review_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", split="full", trust_remote_code=True)

image_dir = "../data/images"
available_images = set(f.split(".")[0] for f in os.listdir(image_dir) if f.endswith(".jpg"))

combined_data = []

for item in review_dataset:
    asin = item.get("parent_asin")
    if asin in available_images:
        title = item.get("title", "")
        description = item.get("text", "")
        
        # Construct full product text
        full_text = " ".join([title, description]).strip()

        combined_data.append({
            "text": full_text,
            "image_path": f"{image_dir}/{asin}.jpg"
        })

with open("../data/combined_data.json", "w", encoding="utf-8") as f:
    json.dump(combined_data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(combined_data)} items to data/combined_data.json")