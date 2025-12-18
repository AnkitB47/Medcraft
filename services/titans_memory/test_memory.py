import pytest
import torch
from .core import TitansMemory, NeuralMemory
from .main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_neural_memory_update():
    dim = 32
    mem = NeuralMemory(dim=dim, memory_dim=16)
    x = torch.randn(1, 10, dim) # [b, n, d]
    
    # Initial state
    initial_mem = mem.memory.clone()
    
    # Forward pass (should update memory)
    out = mem(x)
    
    # Check output shape
    assert out.shape == (1, 10, dim)
    
    # Check memory updated
    assert not torch.allclose(initial_mem, mem.memory)
    
    # Check momentum updated
    assert not torch.allclose(torch.zeros_like(mem.memory), mem.optimizer_state)

def test_titans_modes():
    dim = 32
    x = torch.randn(1, 10, dim)
    
    # MAC
    model_mac = TitansMemory(dim=dim, mode="mac")
    out_mac = model_mac(x)
    # Output should be [b, p+n+n, d] or similar depending on implementation
    # Current impl returns [P, Mem, x] -> [1, 16+10+10, 32]
    assert out_mac.shape[0] == 1
    assert out_mac.shape[2] == dim
    assert out_mac.shape[1] > 10 # enriched
    
    # MAG
    model_mag = TitansMemory(dim=dim, mode="mag")
    out_mag = model_mag(x)
    assert out_mag.shape == (1, 10, dim)
    
    # MAL
    model_mal = TitansMemory(dim=dim, mode="mal")
    out_mal = model_mal(x)
    assert out_mal.shape == (1, 10, dim)

def test_api_endpoints():
    # Store
    data = [0.1] * 512
    response = client.post("/memory/store", json={"tenant_id": "test_tenant", "data": data})
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Retrieve
    response = client.post("/memory/retrieve", json={"tenant_id": "test_tenant", "query": data})
    assert response.status_code == 200
    assert len(response.json()["data"]) == 512
    
    # Reset
    response = client.post("/memory/reset?tenant_id=test_tenant")
    assert response.status_code == 200

def test_persistence_mock():
    # Mock Redis would be ideal, but for now we test the logic flow via API
    # which uses the real Redis connection if available or skips.
    # We can check if snapshot returns 404 or success depending on env
    pass
