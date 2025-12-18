import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms
from PIL import Image

class HybridVisionModel(nn.Module):
    """
    Hybrid Vision Model:
    1. YOLOv8 for Object Detection (Localization)
    2. ViT for Fine-grained Classification of ROIs
    """
    def __init__(self, yolo_path="yolov8n.pt", num_classes=2):
        super().__init__()
        
        # 1. YOLOv8 (Detector)
        # We wrap it but typically run it in inference mode to get boxes
        try:
            self.detector = YOLO(yolo_path)
        except:
            print("Warning: YOLO weights not found, using default")
            self.detector = YOLO("yolov8n.pt")
            
        # 2. ViT (Classifier)
        # We use a pretrained ViT and replace the head for our specific task (e.g. Parkinson vs Normal)
        self.classifier = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        
        # Replace head
        in_features = self.classifier.heads.head.in_features
        self.classifier.heads.head = nn.Linear(in_features, num_classes)
        
        # Preprocessing for ViT
        self.vit_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, image_path_or_array):
        """
        Full pipeline: Detect -> Crop -> Classify
        """
        # 1. Run Detection
        results = self.detector(image_path_or_array)
        
        detections = []
        
        # For each detection, crop and classify
        for r in results:
            boxes = r.boxes
            orig_img = r.orig_img # numpy array
            
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Crop ROI
                roi = orig_img[int(y1):int(y2), int(x1):int(x2)]
                
                if roi.size == 0:
                    continue
                    
                # Convert to PIL for transform
                roi_pil = Image.fromarray(roi)
                
                # Preprocess for ViT
                roi_tensor = self.vit_transform(roi_pil).unsqueeze(0) # [1, 3, 224, 224]
                
                # Run Classification
                with torch.no_grad():
                    logits = self.classifier(roi_tensor)
                    probs = torch.softmax(logits, dim=1)
                    cls_idx = torch.argmax(probs).item()
                    confidence = probs[0][cls_idx].item()
                
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "yolo_class": int(box.cls[0].item()),
                    "vit_class": cls_idx,
                    "vit_confidence": confidence
                })
                
        return detections
