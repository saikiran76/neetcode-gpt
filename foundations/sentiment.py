import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.l1 = nn.Embedding(vocabulary_size, 16)
        self.l2 = nn.Linear(16, 1)
        self.l3 = nn.Sigmoid()
    

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer
        x = self.l1(x)
        x = x.mean(dim=1)
        x = self.l2(x)
        x = self.l3(x)

        # Return a B, 1 tensor and round to 4 decimal places
        return torch.round(x, decimals=4)
