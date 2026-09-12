import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        h = np.array(x)
        for i in range(len(weights)):
            h = np.dot(h, weights[i]) + biases[i]
            if i < len(weights) - 1:      # hidden layers only
                h = np.maximum(0, h)      # ReLU
        return np.round(h, 5)
