import json
import os
import time
import torch
from services.vision_yolo.model import HybridVisionModel
import glob

def eval_yolo(data_dir="data/images"):
    print("Evaluating YOLO/Hybrid Vision...")
    
    # Check data
    image_paths = glob.glob(os.path.join(data_dir, "*.jpg")) + glob.glob(os.path.join(data_dir, "*.png"))
    if not image_paths:
        # Check if we are in a CI environment or just empty
        # User said: "If data is missing → fail loudly"
        raise FileNotFoundError(f"No images found in {data_dir}. Please populate 'data/images' with test data.")
        
    # Load model
    model = HybridVisionModel()
    
    # Metrics
    total_latency = 0
    total_detections = 0
    num_images = len(image_paths)
    
    # Warmup
    print("Warming up...")
    if num_images > 0:
        model.predict(image_paths[0])
        
    print(f"Running evaluation on {num_images} images...")
    for img_path in image_paths:
        start_time = time.time()
        detections = model.predict(img_path)
        end_time = time.time()
        
        total_latency += (end_time - start_time)
        total_detections += len(detections)
        
    avg_latency = total_latency / num_images
    avg_detections = total_detections / num_images
    
    metrics = {
        "num_images": num_images,
        "avg_latency_s": avg_latency,
        "avg_detections_per_image": avg_detections,
        "total_detections": total_detections
    }
    
    os.makedirs("eval/results", exist_ok=True)
    with open("eval/results/yolo_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"YOLO metrics saved: {metrics}")

if __name__ == "__main__":
    # Ensure data dir exists or fail
    if not os.path.exists("data/images"):
        os.makedirs("data/images", exist_ok=True)
        
    try:
        eval_yolo()
    except FileNotFoundError as e:
        print(f"Evaluation failed: {e}")
        # We don't exit 1 here to allow other evals to run in a pipeline if needed, 
        # but for strictness we probably should. 
        # The user said "Fail loudly".
        exit(1)
