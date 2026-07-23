import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        with torch.no_grad():
            current_x = x
            for layer in model:
                current_x = layer(current_x)
                if isinstance(layer, nn.Linear):
                    mean = current_x.mean().item()
                    std = current_x.std().item()

                    is_dead = (current_x <= 0).all(dim=0)
                    dead_frac = is_dead.float().mean().item()
                    
                    stats.append(
                        {
                            'mean': round(mean, 4),
                            'std': round(std, 4),
                            'dead_fraction': round(dead_frac, 4)

                        }
                    )
        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        stats = []
        
        # Call model.zero_grad() first to clear any existing gradients
        model.zero_grad()
        
        # Forward pass
        predictions = model(x)
        
        # Compute loss and backward pass
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, y)
        loss.backward()
        
        # Extract gradient stats for nn.Linear layers
        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                if grad is not None:
                    mean_val = grad.mean().item()
                    std_val = grad.std().item()
                    norm_val = torch.norm(grad).item()
                    
                    stats.append({
                        'mean': round(mean_val, 4),
                        'std': round(std_val, 4),
                        'norm': round(norm_val, 4)
                    })
                    
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # 1. 'dead_neurons' if any layer has dead_fraction > 0.5
        for stat in activation_stats:
            if stat['dead_fraction'] > 0.5:
                return 'dead_neurons'
                
        # 2. 'exploding_gradients' if any layer gradient norm > 1000
        for stat in gradient_stats:
            if stat['norm'] > 1000:
                return 'exploding_gradients'
                
        # 3. 'vanishing_gradients' if last layer gradient norm < 1e-5
        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'
            
        # 4. Check activation std for all layers
        for stat in activation_stats:
            if stat['std'] < 0.1:
                return 'vanishing_gradients'
            if stat['std'] > 10.0:
                return 'exploding_gradients'
                
        # 5. 'healthy' if none of the above conditions are met
        return 'healthy'
