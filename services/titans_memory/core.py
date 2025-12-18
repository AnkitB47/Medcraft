import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralMemory(nn.Module):
    """
    Neural Memory module for Titans.
    Uses a weight matrix M that is updated via gradient descent on a 'surprise' loss.
    """
    def __init__(self, dim, memory_dim=128, learning_rate=0.01, momentum=0.9):
        super().__init__()
        self.dim = dim
        self.memory_dim = memory_dim
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # The memory is a weight matrix M [memory_dim, dim]
        # We treat it as a parameter but update it manually during forward pass
        self.register_buffer('memory', torch.randn(memory_dim, dim))
        self.register_buffer('optimizer_state', torch.zeros_like(self.memory))
        
        # Projections for Key, Value, Query
        self.to_k = nn.Linear(dim, memory_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_q = nn.Linear(dim, memory_dim, bias=False)

    def forward(self, x):
        """
        x: [batch, seq_len, dim]
        """
        b, n, d = x.shape
        
        # 1. Retrieve (Attention over Memory)
        # Q comes from input x
        q = self.to_q(x) # [b, n, memory_dim]
        
        # K, V come from the Memory weights M
        # In Titans, M serves as the weights. 
        # We can interpret M as a collection of key-value associations or just parameters.
        # Simplified interpretation: M * x is the retrieval? 
        # Paper: "The memory module is a neural network... trained online."
        # Let's implement the "Memory as Context" via retrieval:
        # Retrieval = Attention(Q=x, K=M_keys, V=M_values)
        # But M is a matrix. Let's assume M projects input to output.
        
        # Let's stick to the reference implementation style:
        # M is used to process the input.
        # Surprise = || M(k) - v ||^2
        
        # Generate Keys and Values from input for the update
        k = self.to_k(x) # [b, n, memory_dim]
        v = self.to_v(x) # [b, n, dim]
        
        # RETRIEVAL: Use current Memory M to predict v given k
        # prediction = k @ M
        # M shape: [memory_dim, dim]
        # k shape: [b, n, memory_dim]
        # pred shape: [b, n, dim]
        
        # We use the batch-specific memory if we were doing per-sample memory, 
        # but here we have a global shared memory for the tenant (loaded in buffer).
        
        memory_weight = self.memory # [memory_dim, dim]
        retrieved = torch.matmul(k, memory_weight) # [b, n, dim]
        
        # 2. Update (Gradient Descent on Surprise)
        # Loss = MSE(retrieved, v)
        # We want to update M such that M(k) is closer to v.
        # Gradients w.r.t M: (M k - v)^T k  (simplified)
        
        # We need to detach M for the "surprise" calculation to avoid backprop through time if not needed,
        # but for the update we need the gradient.
        
        # Error signal
        error = retrieved - v # [b, n, dim]
        
        # Gradient: dL/dM = k^T * error
        # k: [b, n, memory_dim] -> [b, memory_dim, n]
        # error: [b, n, dim]
        # grad: [b, memory_dim, dim]
        
        grad = torch.matmul(k.transpose(1, 2), error)
        
        # Average grad over batch? Or sum?
        grad = grad.mean(dim=0) # [memory_dim, dim]
        
        # Momentum update
        self.optimizer_state = self.momentum * self.optimizer_state + (1 - self.momentum) * grad
        
        # Apply update
        # M_new = M_old - lr * state
        self.memory = self.memory - self.learning_rate * self.optimizer_state
        
        # Return the retrieved content (memory's "thought" about the input)
        return retrieved

class TitansMemory(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.neural_memory = NeuralMemory(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [b, n, dim]
        memory_out = self.neural_memory(x)
        
        # Gating: Combine Input and Memory
        combined = torch.cat([x, memory_out], dim=-1)
        g = self.gate(combined)
        
        out = g * x + (1 - g) * memory_out
        return out

