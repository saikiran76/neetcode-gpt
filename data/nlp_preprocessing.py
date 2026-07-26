import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them lexicographically, assign integer IDs starting at 1
        all_sentences = positive + negative
        words = set()
        for sentence in all_sentences:
            words.update(sentence.split())
        
        vocab = {word: float(i + 1) for i, word in enumerate(sorted(words))}
        
        # 2. Encode each sentence by replacing words with their IDs
        tensors = []
        for sentence in all_sentences:
            tokens = [vocab[word] for word in sentence.split()]
            tensors.append(torch.tensor(tokens, dtype=torch.float32))
            
        # 3 & 4. Pad shorter sequences with 0s into a rectangular tensor of shape (2N, T)
        return nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0)