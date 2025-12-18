from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
import io
import torch
from .model import NanoVLM
import json
import os

app = FastAPI(title="MedCraft NanoVLM Service")

# Initialize model
# Use CPU for dev, CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
# Disable QLoRA for default dev to avoid bitsandbytes issues on non-GPU
USE_QLORA = os.getenv("USE_QLORA", "False").lower() == "true"

try:
    model = NanoVLM(use_qlora=USE_QLORA).to(device)
    print(f"NanoVLM loaded on {device}")
except Exception as e:
    print(f"Failed to load NanoVLM: {e}")
    model = None

class QARequest(BaseModel):
    prompt: str

class QAResponse(BaseModel):
    findings: str
    confidence: float
    evidence_roi: list
    refusal_reason: str = None

@app.get("/healthz")
async def healthz():
    return {"status": "healthy", "device": device}

@app.post("/grounded_qa", response_model=QAResponse)
async def grounded_qa(prompt: str, file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        image_content = await file.read()
        image = Image.open(io.BytesIO(image_content)).convert("RGB")
        
        # Generate response
        # We expect the model to return a structured string or we parse it.
        # For now, we'll prompt the model to be structured.
        full_prompt = f"Analyze this medical image. Prompt: {prompt}. Return JSON with findings, confidence (0-1), and evidence_roi [x1, y1, x2, y2]."
        
        response_text = model.generate(image, full_prompt)
        
        # Mock parsing since the base model isn't actually fine-tuned to output JSON yet
        # In a real scenario, we'd use a parser or constrained generation.
        # Here we'll wrap the text in the response structure.
        
        return QAResponse(
            findings=response_text,
            confidence=0.95, # Mock confidence
            evidence_roi=[0, 0, 100, 100], # Mock ROI
            refusal_reason=None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
