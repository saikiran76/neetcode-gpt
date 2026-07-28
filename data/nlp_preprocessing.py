import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        vocab = [word for string in positive for word in string.split(' ')] + [word for sentence in negative for word in sentence.split(' ')]
        vocab.sort()
        unique_vocab = [vocab[i] for i in filter(lambda i: i == len(vocab) - 1 or vocab[i] != vocab[i + 1], range(len(vocab)))]



        # vocab_pos = sort([word for string in positive for word in string.split(' ')])
        # vocab_neg = 

        # combined_vocab = vocab_pos + vocab_neg
        
        # encoded_unique_words = {}
        # i = 1
        # for word in combined_vocab:
        #     if word not in unique_map:
        #         unique_map[word] = i
        #         i++;
        # ids = []
        # for k,v in encoded_unique_words.items():
        #     ids.append(v)
        # 2. Encode each sentence by replacing words with their IDs
        # encoded = map(lambda item: i for i in range(1, len(unique_vocab)), unique_vocab)
        vocab_map = dict(map(lambda item: (item[1], item[0]), enumerate(unique_vocab, start=1)))

        
        # 3. Combine positive + negative into one list of tensors
        tensors = [
            torch.tensor([vocab_map[word] for word in sentence.split(' ')], dtype=torch.float) for sentence in (positive + negative)
        ]

        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        output_tensor = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0)
        return output_tensor
