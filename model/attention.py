import torch
import torch.nn as nn
from torchtyping import TensorType
import math

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.attention_dim = attention_dim
        
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        K = self.key(embedded)
        Q = self.query(embedded)
        V = self.value(embedded)

        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # We use .transpose(-2, -1) to swap the context_length and attention_dim dimensions
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.attention_dim)

        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        seq_length = embedded.shape[1]
        mask = torch.tril(torch.ones(seq_length, seq_length, device=embedded.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # 4. Apply softmax(dim=2) to masked scores
        scores = torch.softmax(scores, dim=2)

        # 5. Return (scores @ V) rounded to 4 decimal places
        output = scores @ V
        return torch.round(output, decimals=4)