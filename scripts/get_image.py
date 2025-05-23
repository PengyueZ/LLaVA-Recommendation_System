import os
import json
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Parameters
META_PATH = "../data/meta_All_Beauty.jsonl"
SAVE_DIR = "../data/images"
LOG_PATH = "../data/failed_downloads.txt"
MAX_WORKERS = 10

# Create save directory
os.makedirs(SAVE_DIR, exist_ok=True)

# Load metadata
def load_meta_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            data.append(record)
    return pd.DataFrame(data)

def extract_main_image(images_dict):
    if not isinstance(images_dict, dict):
        return None

    # Try hi_res
    hi_res = images_dict.get('hi_res', [])
    for url in hi_res:
        if url:  # Skip None or empty strings
            return url

    # Fallback to large
    large = images_dict.get('large', [])
    for url in large:
        if url:
            return url

    # Fallback to thumb
    thumb = images_dict.get('thumb', [])
    for url in thumb:
        if url:
            return url
    return None

meta_df = load_meta_jsonl(META_PATH)
meta_df = meta_df[meta_df['images'].notnull()]
meta_df = meta_df[meta_df['images'].apply(lambda x: len(x) > 0)]
meta_df['main_image_url'] = meta_df['images'].apply(extract_main_image)
meta_df = meta_df[meta_df['main_image_url'].notnull()]

print(f"Total rows to process: {len(meta_df)}")

# Download function with retry
def download_image(asin, url, retries=3):
    save_path = os.path.join(SAVE_DIR, f"{asin}.jpg")
    if os.path.exists(save_path):
        return None  # already downloaded
    for _ in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img.save(save_path)
                return None
        except Exception:
            continue
    return asin  # return failed asin

# Multithreaded download
failed_asins = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(download_image, row['parent_asin'], row['main_image_url']): row['parent_asin']
        for _, row in meta_df.iterrows()
    }
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        if result is not None:
            failed_asins.append(result)

# Save failed downloads
with open(LOG_PATH, 'w') as f:
    for asin in failed_asins:
        f.write(f"{asin}\n")

f"✅ Done: {len(meta_df)} products in total. Successfully downloaded {len(meta_df) - len(failed_asins)} image. Failed {len(failed_asins)} images. Failed download stored in {LOG_PATH}"