from ultralytics import YOLO
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
import torch.nn as nn
from PIL import Image
import os

class HybridVisionModel:
    def __init__(self, yolo_path="yolov8n.pt", vit_path="google/vit-base-patch16-224"):
        # 1. Detection (YOLOv8)
        self.yolo = YOLO(yolo_path)
        
        # 2. Classification (ViT)
        self.vit = ViTForImageClassification.from_pretrained(vit_path)
        self.vit_processor = ViTImageProcessor.from_pretrained(vit_path)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vit.to(self.device)

    def predict(self, image_path: str, conf_threshold=0.25):
        # 1. Run YOLO Detection
        results = self.yolo(image_path, conf=conf_threshold)
        
        detections = []
        image = Image.open(image_path).convert("RGB")
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract ROI
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                roi = image.crop((x1, y1, x2, y2))
                
                # 2. Run ViT Classification on ROI
                inputs = self.vit_processor(images=roi, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.vit(**inputs)
                    logits = outputs.logits
                    predicted_class_idx = logits.argmax(-1).item()
                    label = self.vit.config.id2label[predicted_class_idx]
                    confidence = torch.softmax(logits, dim=-1)[0, predicted_class_idx].item()
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "yolo_class": r.names[int(box.cls[0])],
                    "yolo_conf": float(box.conf[0]),
                    "vit_class": label,
                    "vit_conf": confidence,
                    "hybrid_conf": (float(box.conf[0]) + confidence) / 2
                })
                
        return detections

    def export_onnx(self, output_path="models/hybrid_vision.onnx"):
        # Export YOLO
        self.yolo.export(format="onnx")
        
        # Export ViT
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        torch.onnx.export(
            self.vit, 
            dummy_input, 
            "models/vit.onnx", 
            input_names=["pixel_values"], 
            output_names=["logits"]
        )
        print("Models exported to ONNX")
