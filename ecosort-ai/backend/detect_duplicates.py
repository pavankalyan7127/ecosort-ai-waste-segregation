"""Duplicate and near-duplicate detection using SHA-256 and Perceptual dHash."""

import json
from pathlib import Path
from PIL import Image
import numpy as np

images_dir = Path(r"D:\EcoSort-Resources\datasets\TACO\processed\images")

def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Calculate difference hash (dHash) for an image."""
    # Resize to (hash_size + 1, hash_size)
    img = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(img)
    # Compare adjacent pixels
    diff = pixels[:, 1:] > pixels[:, :-1]
    # Convert bool array to 64-bit int
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def hamming_distance(h1: int, h2: int) -> int:
    """Compute bitwise Hamming distance between two 64-bit hashes."""
    return bin(h1 ^ h2).count("1")

print("Computing perceptual and cryptographic hashes for 1,072 images...")

image_files = sorted(list(images_dir.glob("*.jpg")), key=lambda p: int(p.stem))

hashes = {}
import hashlib

for p in image_files:
    img_id = int(p.stem)
    with open(p, "rb") as f:
        bytes_data = f.read()
    sha256 = hashlib.sha256(bytes_data).hexdigest()
    md5 = hashlib.md5(bytes_data).hexdigest()
    with Image.open(p) as img:
        phash = dhash(img)
    hashes[img_id] = {"sha256": sha256, "md5": md5, "dhash": phash, "path": str(p)}

print("Hashes computed. Finding exact and near-duplicates...")

# 1. Exact duplicates by MD5/SHA256
exact_duplicates = []
seen_md5 = {}
for img_id, h in hashes.items():
    md5 = h["md5"]
    if md5 in seen_md5:
        exact_duplicates.append({"image_id": img_id, "duplicate_of": seen_md5[md5], "type": "exact_cryptographic"})
    else:
        seen_md5[md5] = img_id

print(f"Exact cryptographic duplicates: {len(exact_duplicates)}")

# 2. Near duplicates by dHash (Hamming distance <= 4)
near_duplicates = []
img_ids = list(hashes.keys())

for i in range(len(img_ids)):
    id1 = img_ids[i]
    h1 = hashes[id1]["dhash"]
    for j in range(i + 1, len(img_ids)):
        id2 = img_ids[j]
        h2 = hashes[id2]["dhash"]
        dist = hamming_distance(h1, h2)
        if dist <= 4:
            near_duplicates.append({
                "image_id_1": id1,
                "image_id_2": id2,
                "hamming_distance": dist,
                "type": "exact_visual" if dist == 0 else "near_duplicate_burst",
            })

print(f"Near-duplicate pairs (Hamming distance <= 4): {len(near_duplicates)}")
for dup in near_duplicates:
    print(" ", dup)

# Build duplicate clusters / groups for splitting
# Connected components graph
parent = {img_id: img_id for img_id in img_ids}

def find(i):
    if parent[i] == i:
        return i
    parent[i] = find(parent[i])
    return parent[i]

def union(i, j):
    root_i = find(i)
    root_j = find(j)
    if root_i != root_j:
        parent[root_i] = root_j

for pair in near_duplicates:
    union(pair["image_id_1"], pair["image_id_2"])

clusters = {}
for img_id in img_ids:
    root = find(img_id)
    clusters.setdefault(root, []).append(img_id)

multi_image_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
print(f"\nTotal duplicate/near-duplicate clusters (>1 image): {len(multi_image_clusters)}")
for root, members in multi_image_clusters.items():
    print(f" Cluster root {root}: {members}")
