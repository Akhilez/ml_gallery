import torch
from lib.nn_utils import get_scaled_random_weights


class SOFM1D(torch.nn.Module):

    COSINE_OP = "cosine"
    DISTANCE_OP = "distance"

    def __init__(self, input_size, output_size, lr=0.1, operation=DISTANCE_OP):
        super().__init__()
        self.w = get_scaled_random_weights((input_size, output_size), -1, 1)
        self.w.requires_grad = False
        self.lr = lr

    def forward(self, x):
        """
        1. Find distances b/w wi and x.
        2. i* = arg min of distances
        3. Get neuron_distance factor
        4.
        Parameters
        ----------
        x: batch of 1D Tensors
        """
        deltas = []
        y_hat = []
        for xi in x:
            differences = torch.sum((self.w.T - xi) ** 2, dim=1)
            i_min = torch.argmin(differences)
            deltas.append()
            # TODO: Finish SOFM

    def backward(self):
        pass
