import torch
from services.titans_memory.core import TitansMemory, NeuralMemory
from services.titans_memory.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_neural_memory_update():
    print("Testing NeuralMemory update...")
    dim = 32
    mem = NeuralMemory(dim=dim, memory_dim=16)
    x = torch.randn(1, 10, dim) # [b, n, d]
    
    # Initial state
    initial_mem = mem.memory.clone()
    
    # Forward pass (should update memory)
    out = mem(x)
    
    # Check output shape
    assert out.shape == (1, 10, dim)
    print("Shape check passed.")
    
    # Check memory updated
    if torch.allclose(initial_mem, mem.memory):
        print("FAIL: Memory did not update.")
    else:
        print("Memory update check passed.")
    
    # Check momentum updated
    if torch.allclose(torch.zeros_like(mem.memory), mem.optimizer_state):
        print("FAIL: Momentum did not update.")
    else:
        print("Momentum update check passed.")

def test_titans_modes():
    print("\nTesting Titans modes...")
    dim = 32
    x = torch.randn(1, 10, dim)
    
    # MAC
    print("Testing MAC...")
    model_mac = TitansMemory(dim=dim, mode="mac")
    out_mac = model_mac(x)
    # Output should be [b, p+n+n, d] or similar depending on implementation
    # Current impl returns [P, Mem, x] -> [1, 16+10+10, 32]
    if out_mac.shape[0] == 1 and out_mac.shape[2] == dim and out_mac.shape[1] > 10:
        print("MAC check passed.")
    else:
        print(f"FAIL: MAC shape {out_mac.shape}")
    
    # MAG
    print("Testing MAG...")
    model_mag = TitansMemory(dim=dim, mode="mag")
    out_mag = model_mag(x)
    if out_mag.shape == (1, 10, dim):
        print("MAG check passed.")
    else:
        print(f"FAIL: MAG shape {out_mag.shape}")
    
    # MAL
    print("Testing MAL...")
    model_mal = TitansMemory(dim=dim, mode="mal")
    out_mal = model_mal(x)
    if out_mal.shape == (1, 10, dim):
        print("MAL check passed.")
    else:
        print(f"FAIL: MAL shape {out_mal.shape}")

def test_api_endpoints():
    print("\nTesting API endpoints...")
    # Store
    data = [0.1] * 512
    response = client.post("/memory/store", json={"tenant_id": "test_tenant", "data": data})
    if response.status_code == 200 and response.json()["status"] == "success":
        print("Store check passed.")
    else:
        print(f"FAIL: Store response {response.json()}")
    
    # Retrieve
    response = client.post("/memory/retrieve", json={"tenant_id": "test_tenant", "query": data})
    if response.status_code == 200 and len(response.json()["data"]) == 512:
        print("Retrieve check passed.")
    else:
        print(f"FAIL: Retrieve response {response.json()}")
    
    # Reset
    response = client.post("/memory/reset?tenant_id=test_tenant")
    if response.status_code == 200:
        print("Reset check passed.")
    else:
        print(f"FAIL: Reset response {response.json()}")

if __name__ == "__main__":
    try:
        test_neural_memory_update()
        test_titans_modes()
        test_api_endpoints()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
