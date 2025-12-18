import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralMemory(nn.Module):
    """
    Neural Memory module for Titans.
    Uses a weight matrix M that is updated via gradient descent on a 'surprise' loss.
    Updates: M_t = (1 - alpha_t) * M_{t-1} + S_t
    Surprise: S_t = eta_t * S_{t-1} - theta_t * grad(Loss)
    Loss: || M(k) - v ||^2
    """
    def __init__(self, dim, memory_dim=128, learning_rate=0.01, momentum=0.9):
        super().__init__()
        self.dim = dim
        self.memory_dim = memory_dim
        self.learning_rate = learning_rate
        self.default_momentum = momentum
        
        # The memory is a weight matrix M [memory_dim, dim]
        self.register_buffer('memory', torch.randn(memory_dim, dim))
        self.register_buffer('optimizer_state', torch.zeros_like(self.memory)) # S_{t-1}
        
        # Projections for Key, Value, Query
        self.to_k = nn.Linear(dim, memory_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_q = nn.Linear(dim, memory_dim, bias=False)
        
        # Adaptive parameters (alpha, eta, theta)
        # In a full implementation, these would be data-dependent (functions of x_t)
        # For this implementation, we make them learnable scalars or simple functions
        self.alpha_gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid()) # Forgetting
        self.eta_gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())   # Momentum decay
        self.theta_gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid()) # Learning rate modulation

    def forward(self, x):
        """
        x: [batch, seq_len, dim]
        """
        b, n, d = x.shape
        
        # 1. Retrieve (Attention over Memory)
        q = self.to_q(x) # [b, n, memory_dim]
        
        # Retrieval: M * q^T? Or q * M?
        # M maps memory_dim -> dim. 
        # So retrieved = q @ M
        memory_weight = self.memory # [memory_dim, dim]
        retrieved = torch.matmul(q, memory_weight) # [b, n, dim]
        
        # 2. Update (Gradient Descent on Surprise)
        # We need to compute gradients w.r.t M for the loss || M(k) - v ||^2
        k = self.to_k(x) # [b, n, memory_dim]
        v = self.to_v(x) # [b, n, dim]
        
        # We process token by token or chunk by chunk for correct causal updates.
        # For simplicity in this "in-place" implementation, we'll do a simplified batch update 
        # that approximates the online learning. In a strict loop, we'd iterate.
        
        # Let's do a mean update over the sequence to simulate "learning from this context"
        # Prediction using current memory
        pred_v = torch.matmul(k, memory_weight)
        
        # Error
        error = pred_v - v # [b, n, dim]
        
        # Gradient dL/dM = k^T * error
        # [b, n, memory_dim]^T * [b, n, dim] -> [b, memory_dim, dim]
        grad = torch.matmul(k.transpose(1, 2), error).mean(dim=0) # Average over batch
        
        # Adaptive gates (averaged over sequence for this update step)
        x_mean = x.mean(dim=1) # [b, dim]
        alpha = self.alpha_gate(x_mean).mean() # Scalar forgetting
        eta = self.eta_gate(x_mean).mean()     # Scalar momentum decay
        theta = self.theta_gate(x_mean).mean() # Scalar lr modulation
        
        # Update Momentum (Surprise)
        # S_t = eta * S_{t-1} - theta * grad
        self.optimizer_state = eta * self.optimizer_state - theta * grad
        
        # Update Memory
        # M_t = (1 - alpha) * M_{t-1} + S_t
        self.memory = (1 - alpha) * self.memory + self.optimizer_state
        
        return retrieved

class TitansMemory(nn.Module):
    def __init__(self, dim, num_persistent_tokens=16, mode="mac"):
        super().__init__()
        self.mode = mode.lower()
        self.dim = dim
        self.neural_memory = NeuralMemory(dim)
        
        # Persistent Memory (Learnable prefix tokens)
        self.persistent_memory = nn.Parameter(torch.randn(1, num_persistent_tokens, dim))
        
        # Gating for MAG
        if self.mode == "mag":
            self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
            
        # Layers for MAL
        if self.mode == "mal":
            self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, n, d = x.shape
        
        # Prepend persistent memory
        # P: [1, p, d] -> [b, p, d]
        p_mem = self.persistent_memory.expand(b, -1, -1)
        
        # In MAC, we might concatenate P to input for attention, 
        # but here we treat P as part of the context that the memory "sees" or is conditioned on.
        # For simplicity, we'll pass x through memory.
        
        if self.mode == "mac":
            # Memory As Context
            # Retrieve from memory
            mem_out = self.neural_memory(x) # [b, n, d]
            # Concatenate P, Mem, Input? 
            # Usually MAC means Mem is added to context.
            # Here we return [P, Mem_out, x] or just Mem_out depending on usage.
            # Let's return the enriched context.
            return torch.cat([p_mem, mem_out, x], dim=1)
            
        elif self.mode == "mag":
            # Memory As Gating
            mem_out = self.neural_memory(x)
            combined = torch.cat([x, mem_out], dim=-1)
            g = self.gate(combined)
            out = g * x + (1 - g) * mem_out
            return out
            
        elif self.mode == "mal":
            # Memory As Layer
            # Input -> Memory -> Norm -> Output
            mem_out = self.neural_memory(x)
            return self.layer_norm(x + mem_out)
            
        else:
            return x


