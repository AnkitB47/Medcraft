from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
from .model import HybridVisionModel

app = FastAPI(title="MedCraft Vision YOLO Service")

# Initialize model
try:
    model = HybridVisionModel()
    print("HybridVisionModel loaded")
except Exception as e:
    print(f"Failed to load HybridVisionModel: {e}")
    model = None

class Detection(BaseModel):
    bbox: list
    yolo_class: str
    yolo_conf: float
    vit_class: str
    vit_conf: float
    hybrid_conf: float

class AnalysisResponse(BaseModel):
    detections: list[Detection]
    count: int

@app.get("/healthz")
async def healthz():
    return {"status": "healthy"}

@app.post("/analyze_parkinson", response_model=AnalysisResponse)
async def analyze_parkinson(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        # Save temp file
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run inference
        detections = model.predict(temp_file)
        
        # Cleanup
        os.remove(temp_file)
        
        return AnalysisResponse(
            detections=detections,
            count=len(detections)
        )
        
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise HTTPException(status_code=500, detail=str(e))
