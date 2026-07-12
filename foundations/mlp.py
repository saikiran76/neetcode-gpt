import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)x
        z = np.array([])
        layer = x
        for i in range(len(weights)):
            z = np.dot(layer, np.array(weights[i])) + biases[i]
            z = np.maximum(z, np.zeros(z.shape))
            layer = z

        # to form final layer (output)
        

        return z
        
