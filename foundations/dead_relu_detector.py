import torch
import torch.nn as nn
from typing import List

class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions = []

        with torch.no_grad():
            curr = x
            for layer in model:
                curr = layer(curr)
                if isinstance(layer, nn.ReLU):
                    # A neuron is dead if it outputs 0 for ALL samples in the batch.
                    # Flatten spatial dimensions if present (batch_size, num_neurons)
                    out_flat = curr.view(curr.size(0), -1)
                    
                    # Check across batch dimension (dim=0): true if output == 0 for all batch samples
                    is_dead = (out_flat == 0).all(dim=0)
                    
                    # Fraction of dead neurons
                    fraction = is_dead.float().mean().item()
                    dead_fractions.append(round(fraction, 4))

        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        if not dead_fractions:
            return 'healthy'

        # 1. Any layer has dead fraction > 0.5
        if any(df > 0.5 for df in dead_fractions):
            return 'use_leaky_relu'

        # 2. First layer has dead fraction > 0.3
        if dead_fractions[0] > 0.3:
            return 'reinitialize'

        # 3. Strictly increases with depth AND last layer's fraction > 0.1
        is_strictly_increasing = all(
            dead_fractions[i] < dead_fractions[i + 1]
            for i in range(len(dead_fractions) - 1)
        )
        if is_strictly_increasing and dead_fractions[-1] > 0.1:
            return 'reduce_learning_rate'

        # 4. Max dead fraction < 0.1
        if max(dead_fractions) < 0.1:
            return 'healthy'

        # 5. Otherwise
        return 'healthy'