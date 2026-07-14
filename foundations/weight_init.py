import torch
import math
from typing import List

class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        W = torch.randn(fan_out, fan_in) * std
        return torch.round(W, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        W = torch.randn(fan_out, fan_in) * std
        return torch.round(W, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)
        
        # 1. Build ALL weight matrices first to maintain expected random sequence
        weights = []
        for i in range(num_layers):
            fan_in = input_dim if i == 0 else hidden_dim
            fan_out = hidden_dim
            
            if init_type == 'xavier':
                std = math.sqrt(2.0 / (fan_in + fan_out))
            elif init_type == 'kaiming':
                std = math.sqrt(2.0 / fan_in)
            else:  # 'random'
                std = 1.0
                
            W = torch.randn(fan_out, fan_in) * std
            weights.append(W)
            
        # 2. Generate the random input vector AFTER the weights
        x = torch.randn(1, input_dim) 
        
        # 3. Forward pass through the network
        stds = []
        for W in weights:
            # Linear transformation + ReLU activation
            x = x @ W.T
            x = torch.relu(x)
            
            # Record the standard deviation rounded to 2 decimals
            stds.append(round(x.std().item(), 2))
            
        return stds