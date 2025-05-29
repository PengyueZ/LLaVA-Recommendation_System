from transformers import CLIPProcessor, CLIPModel
import json
import torch
from PIL import Image
import os
import numpy as np
import faiss
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

with open("../data/combined_data.json", "r", encoding="utf-8") as f:
    sample_data = json.load(f)

def get_embedding(text, image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True, truncation = True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    text_emb = outputs.text_embeds[0]
    image_emb = outputs.image_embeds[0]
    combined_emb = torch.cat([image_emb, text_emb], dim=-1)  # Concatenation
    return combined_emb / combined_emb.norm()

embeddings = []
for item in sample_data:
    emb = get_embedding(item["text"], item["image_path"])
    embeddings.append(emb.cpu().numpy())

embedding_matrix = np.vstack(embeddings).astype("float32")

faiss_index = faiss.IndexFlatIP(embedding_matrix.shape[1])
faiss_index.add(embedding_matrix)

# Query example (text + image)
query_text = "A hydrating product for dry skin"
query_image = "../data/OOO5946468.jpg"  # image path from your data

query_emb = get_embedding(query_text, query_image).cpu().numpy().reshape(1, -1)
D, I = faiss_index.search(query_emb, k=3)

# Show results
for idx in I[0]:
    print(f"Match: {sample_data[idx]['text']}")
    img = Image.open(sample_data[idx]['image_path'])
    plt.imshow(img)
    plt.axis('off')  # Hide axes for better visualization
    plt.show()