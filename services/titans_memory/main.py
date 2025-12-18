from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from .core import TitansMemory
import os
import redis
import pickle
import base64

app = FastAPI(title="MedCraft Titans Memory Service")

# Configuration
DIM = 512
TITANS_MODE = os.getenv("TITANS_MODE", "mac") # mac, mag, mal
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Initialize model
model = TitansMemory(dim=DIM, mode=TITANS_MODE)

# Redis Connection
try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
except:
    print("Warning: Redis not connected. Persistence disabled.")
    r = None

class MemoryRequest(BaseModel):
    tenant_id: str
    data: list # List of floats [dim] or [seq, dim]

class MemoryRetrieveRequest(BaseModel):
    tenant_id: str
    query: list # List of floats [dim] or [seq, dim]

def load_memory_state(tenant_id: str):
    if not r: return None
    state = r.get(f"titans_memory:{tenant_id}")
    if state:
        return pickle.loads(state)
    return None

def save_memory_state(tenant_id: str, state):
    if not r: return
    r.set(f"titans_memory:{tenant_id}", pickle.dumps(state))

@app.get("/healthz")
async def healthz():
    return {"status": "healthy", "mode": TITANS_MODE}

@app.post("/memory/store")
async def store_memory(request: MemoryRequest):
    try:
        # Input shape handling
        x = torch.tensor(request.data)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0) # [1, 1, dim]
        elif x.dim() == 2:
            x = x.unsqueeze(0) # [1, seq, dim]
            
        if x.shape[-1] != DIM:
            raise HTTPException(status_code=400, detail=f"Input dimension must be {DIM}")
        
        # Load state
        state = load_memory_state(request.tenant_id)
        if state:
            model.neural_memory.load_state_dict(state)
        
        # Forward pass (updates memory internally via NeuralMemory)
        # We don't need the output for storage, just the side effect of update
        with torch.no_grad(): # We manually update weights, so no autograd needed for the update mechanism itself? 
            # Actually, our NeuralMemory implementation does manual updates in forward.
            # But wait, if we use `model(x)`, it returns output.
            # The update happens in `neural_memory.forward`.
            model(x)
        
        # Save state
        save_memory_state(request.tenant_id, model.neural_memory.state_dict())
        
        return {"status": "success", "message": "Memory updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/retrieve")
async def retrieve_memory(request: MemoryRetrieveRequest):
    try:
        x = torch.tensor(request.query)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(0)
            
        if x.shape[-1] != DIM:
            raise HTTPException(status_code=400, detail=f"Input dimension must be {DIM}")
        
        # Load state
        state = load_memory_state(request.tenant_id)
        if state:
            model.neural_memory.load_state_dict(state)
            
        # Forward pass to retrieve
        with torch.no_grad():
            out = model(x)
            
        return {"status": "success", "data": out.squeeze().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/reset")
async def reset_memory(tenant_id: str):
    if r:
        r.delete(f"titans_memory:{tenant_id}")
    return {"status": "success", "message": f"Memory reset for tenant {tenant_id}"}

@app.post("/memory/snapshot")
async def snapshot_memory(tenant_id: str):
    if not r:
        return {"status": "error", "message": "Redis not configured"}
        
    state = r.get(f"titans_memory:{tenant_id}")
    if not state:
        raise HTTPException(status_code=404, detail="No memory found for tenant")
    
    snapshot = base64.b64encode(state).decode('utf-8')
    return {"status": "success", "snapshot": snapshot}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
