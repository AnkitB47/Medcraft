import torch
import pytest
from services.titans_memory.core import MemoryAsContext, MemoryAsGating, MemoryAsLayer, TitansMemory

def test_mac():
    dim = 128
    num_tokens = 5
    mac = MemoryAsContext(dim, num_memory_tokens=num_tokens)
    x = torch.randn(2, 10, dim)
    out = mac(x)
    assert out.shape == (2, 10 + num_tokens, dim)

def test_mag():
    dim = 128
    mag = MemoryAsGating(dim)
    x = torch.randn(2, 10, dim)
    out = mag(x)
    assert out.shape == (2, 10, dim)

def test_mal():
    dim = 128
    mal = MemoryAsLayer(dim)
    x = torch.randn(2, 10, dim)
    out = mal(x)
    assert out.shape == (2, 10, dim)
    
    # Check if memory updates (simple check)
    initial_memory = mal.memory.clone()
    mal(x)
    assert not torch.equal(initial_memory, mal.memory)

def test_titans_memory():
    dim = 128
    model = TitansMemory(dim)
    x = torch.randn(2, 10, dim)
    out = model(x)
    # MAC adds tokens, MAG and MAL keep shape
    assert out.shape == (2, 10 + 10, dim) # Default num_memory_tokens is 10
