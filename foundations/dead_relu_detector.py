import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead_fracs = []
        with torch.no_grad():
            curr = x
            for layer in model:
                curr = layer(curr)
                if isinstance(layer, nn.ReLU):
                    # out_flat = curr.view(curr.size(0), -1)
                    is_dead_layer = (curr == 0).all(dim=0)
                    dead_frac = is_dead_layer.float().mean().item()
                    dead_fracs.append(round(dead_frac, 4))
        return dead_fracs




    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        if not dead_fractions:
            return 'healthy'
        if any(df > 0.5 for df in dead_fractions):
            return 'use_leaky_relu'
        if dead_fractions[0] > 0.3:
            return 'reinitialize'
        is_increasing = all(
            dead_fractions[i] < dead_fractions[i+1] for i in range(len(dead_fractions)-1)
        )
        if is_increasing:
            return 'reduce_learning_rate'
        if max(dead_fractions) < 0.1:
            return 'healthy'
        return 'healthy'

