import os
import json
from PIL import Image, ImageDraw
import random

def setup_dev_data():
    print("Setting up dev data...")
    
    # 1. Images
    os.makedirs("data/images", exist_ok=True)
    img_path = "data/images/sample.jpg"
    if not os.path.exists(img_path):
        print(f"Creating dummy image at {img_path}...")
        img = Image.new('RGB', (224, 224), color = (73, 109, 137))
        d = ImageDraw.Draw(img)
        d.text((10,10), "MedCraft Sample", fill=(255,255,0))
        img.save(img_path)
        
    # 2. JSONL Data for Finetuning
    jsonl_path = "data/curated_feedback.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"Creating dummy dataset at {jsonl_path}...")
        sample_data = {
            "image_path": img_path,
            "prompt": "What is in this image?",
            "answer": "This is a sample medical image for testing purposes.",
            "roi_ids": [1]
        }
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(sample_data) + "\n")
            
    print("Dev data setup complete.")

if __name__ == "__main__":
    setup_dev_data()
