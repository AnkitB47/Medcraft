import json
import os
import time
import torch
from services.nanovlm.model import NanoVLM
from PIL import Image
import glob

def eval_nanovlm(data_dir="data/images"):
    print("Evaluating NanoVLM...")
    
    # Check data
    image_paths = glob.glob(os.path.join(data_dir, "*.jpg")) + glob.glob(os.path.join(data_dir, "*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {data_dir}. Please populate 'data/images' with test data.")
        
    # Load model
    # Use CPU for eval if GPU not available, but prefer GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading NanoVLM on {device}...")
    model = NanoVLM(use_qlora=False).to(device) # No QLoRA for fast eval unless needed
    
    # Metrics
    total_latency = 0
    consistency_score = 0
    num_images = len(image_paths)
    prompt = "Describe this medical image."
    
    # Warmup
    print("Warming up...")
    if num_images > 0:
        img = Image.open(image_paths[0]).convert("RGB")
        model.generate(img, prompt)
        
    print(f"Running evaluation on {num_images} images...")
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        
        # Run 1
        start_time = time.time()
        res1, conf1 = model.generate(img, prompt)
        end_time = time.time()
        total_latency += (end_time - start_time)
        
        # Run 2 (Consistency)
        res2, conf2 = model.generate(img, prompt)
        
        # Simple consistency: exact match or high overlap
        if res1 == res2:
            consistency_score += 1
            
    avg_latency = total_latency / num_images
    avg_consistency = consistency_score / num_images
    
    metrics = {
        "num_images": num_images,
        "avg_latency_s": avg_latency,
        "consistency_score": avg_consistency,
        "device": device
    }
    
    os.makedirs("eval/results", exist_ok=True)
    with open("eval/results/nanovlm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"NanoVLM metrics saved: {metrics}")

if __name__ == "__main__":
    if not os.path.exists("data/images"):
        os.makedirs("data/images", exist_ok=True)
        
    try:
        eval_nanovlm()
    except FileNotFoundError as e:
        print(f"Evaluation failed: {e}")
        exit(1)
