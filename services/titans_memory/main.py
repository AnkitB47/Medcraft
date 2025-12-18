from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from .core import TitansMemory
import os
import redis
import pickle
import base64

app = FastAPI(title="MedCraft Titans Memory Service")

# Initialize model
DIM = 512
model = TitansMemory(dim=DIM)

# Redis Connection
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False) # Binary for pickle

class MemoryStoreRequest(BaseModel):
    data: list # List of floats (embeddings)
    tenant_id: str

class MemoryRetrieveRequest(BaseModel):
    query: list # List of floats (embeddings)
    tenant_id: str

def load_memory_state(tenant_id: str):
    state = r.get(f"titans_memory:{tenant_id}")
    if state:
        return pickle.loads(state)
    return None

def save_memory_state(tenant_id: str, state):
    r.set(f"titans_memory:{tenant_id}", pickle.dumps(state))

@app.get("/healthz")
async def healthz():
    return {"status": "healthy"}

@app.post("/memory/store")
async def store_memory(request: MemoryStoreRequest):
    try:
        x = torch.tensor(request.data).unsqueeze(0).unsqueeze(0) # [1, 1, dim]
        if x.shape[-1] != DIM:
            raise HTTPException(status_code=400, detail=f"Input dimension must be {DIM}")
        
        # Load state
        state = load_memory_state(request.tenant_id)
        if state:
            model.load_state_dict(state)
        
        # Update memory (forward pass updates internal state in MAL)
        model(x)
        
        # Save state
        save_memory_state(request.tenant_id, model.state_dict())
        
        return {"status": "success", "message": "Memory updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/retrieve")
async def retrieve_memory(request: MemoryRetrieveRequest):
    try:
        x = torch.tensor(request.query).unsqueeze(0).unsqueeze(0)
        if x.shape[-1] != DIM:
            raise HTTPException(status_code=400, detail=f"Input dimension must be {DIM}")
        
        # Load state
        state = load_memory_state(request.tenant_id)
        if state:
            model.load_state_dict(state)
            
        out = model(x)
        return {"status": "success", "data": out.squeeze().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/reset")
async def reset_memory(tenant_id: str):
    r.delete(f"titans_memory:{tenant_id}")
    return {"status": "success", "message": f"Memory reset for tenant {tenant_id}"}

@app.post("/memory/snapshot")
async def snapshot_memory(tenant_id: str):
    state = r.get(f"titans_memory:{tenant_id}")
    if not state:
        raise HTTPException(status_code=404, detail="No memory found for tenant")
    
    # Encrypt snapshot (mock encryption)
    snapshot = base64.b64encode(state).decode('utf-8')
    return {"status": "success", "snapshot": snapshot}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
