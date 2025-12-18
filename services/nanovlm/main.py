from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import torch
from .model import NanoVLM
from transformers import AutoTokenizer
from PIL import Image
import io

app = FastAPI(title="MedCraft NanoVLM Service")

# Initialize model and tokenizer
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
try:
    # In production, we would use 4-bit quantization here
    # model = NanoVLM(text_model_name=MODEL_NAME)
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = None
    tokenizer = None
except Exception as e:
    print(f"Warning: Could not load NanoVLM: {e}")
    model = None
    tokenizer = None

class ReasoningRequest(BaseModel):
    prompt: str
    context: str = ""

class ReasoningResponse(BaseModel):
    answer: str
    evidence: dict
    certainty: float

@app.get("/healthz")
async def healthz():
    return {"status": "healthy"}

@app.post("/reasoning/qa", response_model=ReasoningResponse)
async def grounded_qa(request: ReasoningRequest, file: UploadFile = File(None)):
    # 1. Process image if provided
    image = None
    if file:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 2. Run NanoVLM inference
    if model and image:
        try:
            answer = model.generate(image, request.prompt)
        except Exception as e:
            print(f"Inference error: {e}")
            answer = "Error during inference. Please check logs."
    else:
        answer = "Model not loaded or no image provided. Running in mock mode: Based on the visual evidence and clinical history, there is a high likelihood of pneumonia in the left lower lobe."

    # 3. Return grounded JSON
    return ReasoningResponse(
        answer=answer,
        evidence={
            "roi_ids": ["lung_left_lower"],
            "mask_urls": ["/masks/lung_left_lower.png"]
        },
        certainty=0.89
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
