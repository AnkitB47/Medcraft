import torch
import os
import pickle
import redis
from services.titans_memory.core import TitansMemory

# Mock Redis for standalone test if needed, or use real if available
# For this proof, we'll simulate the save/load mechanism directly using the same logic as main.py

def test_persistence():
    print("Testing Titans Persistence...")
    dim = 32
    model = TitansMemory(dim=dim)
    
    # 1. Initial State
    print("Initializing model...")
    initial_memory = model.neural_memory.memory.clone()
    
    # 2. Store (Update)
    print("Updating memory with input A...")
    x = torch.randn(1, 10, dim)
    model(x) # Forward pass updates memory
    
    updated_memory = model.neural_memory.memory.clone()
    
    # Verify update happened
    diff = (updated_memory - initial_memory).abs().sum().item()
    print(f"Memory update diff: {diff}")
    if diff == 0:
        print("FAIL: Memory did not update.")
        return
        
    # 3. Save State (Simulate Redis set)
    print("Saving state...")
    state_bytes = pickle.dumps(model.neural_memory.state_dict())
    
    # 4. Restart (New Instance)
    print("Restarting service (creating new model)...")
    new_model = TitansMemory(dim=dim)
    
    # Verify new model is different from updated model (random init)
    # Actually, buffers are init to randn, so they will differ.
    
    # 5. Load State (Simulate Redis get)
    print("Loading state...")
    state_dict = pickle.loads(state_bytes)
    new_model.neural_memory.load_state_dict(state_dict)
    
    # 6. Retrieve
    print("Retrieving from new model...")
    # Check if memory matches updated_memory
    loaded_memory = new_model.neural_memory.memory
    
    load_diff = (loaded_memory - updated_memory).abs().sum().item()
    print(f"Load diff (should be 0): {load_diff}")
    
    if load_diff < 1e-6:
        print("SUCCESS: Persistence verified. Memory state preserved across instances.")
    else:
        print("FAIL: Persistence failed.")

if __name__ == "__main__":
    test_persistence()
