import torch
from lib.nn_utils import get_scaled_random_weights


class SOFM1D(torch.nn.Module):

    COSINE_OP = 'cosine'
    DISTANCE_OP = 'distance'

    def __init__(self, input_shape, output_shape, operation=DISTANCE_OP):
        super().__init__()
        self.weights = get_scaled_random_weights((input_shape, output_shape), -1, 1)

    def forward(self, x):
        pass  # TODO: Do the operation

