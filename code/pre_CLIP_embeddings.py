import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import faiss
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load dataset
with open("../data/combined_data.json", "r", encoding="utf-8") as f:
    sample_data = json.load(f)

# Dataset class to preload data
class CLIPDataset(Dataset):
    def __init__(self, sample_data):
        self.data = sample_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        image_path = item['image_path']
        image = Image.open(image_path).convert("RGB")
        return text, image
    
# Collate function for batching
def collate_fn(batch):
    texts, images = zip(*batch)
    return list(texts), list(images)

# Create DataLoader
batch_size = 32
dataset = CLIPDataset(sample_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Embedding extraction
all_embeddings = []

for texts, images in tqdm(dataloader, desc="Extracting embeddings"):
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    text_emb = outputs.text_embeds
    image_emb = outputs.image_embeds
    combined_emb = torch.cat([image_emb, text_emb], dim=-1)
    normalized_emb = combined_emb / combined_emb.norm(dim=-1, keepdim=True)

    all_embeddings.append(normalized_emb.cpu())

# Stack everything into one numpy array
embedding_matrix = torch.cat(all_embeddings, dim=0).numpy().astype("float32")

# Build FAISS index
faiss_index = faiss.IndexFlatIP(embedding_matrix.shape[1])
faiss_index.add(embedding_matrix)

# Prepare a query
query_text = "A hydrating product for dry skin"
query_image_path = "../data/OOO5946468.jpg"

# Query embedding (single sample but still batched for consistency)
query_image = Image.open(query_image_path).convert("RGB")
inputs = processor(text=[query_text], images=[query_image], return_tensors="pt", padding=True, truncation=True).to(device)

with torch.no_grad():
    outputs = model(**inputs)

query_text_emb = outputs.text_embeds
query_image_emb = outputs.image_embeds
query_combined_emb = torch.cat([query_image_emb, query_text_emb], dim=-1)
query_combined_emb = query_combined_emb / query_combined_emb.norm(dim=-1, keepdim=True)

query_embedding = query_combined_emb.cpu().numpy().astype("float32")

# Search
D, I = faiss_index.search(query_embedding, k=3)

# Display results
for idx in I[0]:
    print(f"Match: {sample_data[idx]['text']}")
    img = Image.open(sample_data[idx]['image_path'])
    plt.imshow(img)
    plt.axis('off')
    plt.show()