import torch
from datetime import datetime


def get_scaled_random_weights(shape, min_=-0.5, max_=0.5):
    return torch.FloatTensor(*shape).uniform_(min_, max_).requires_grad_()


def save_model(model, cwd, name):
    timestamp = int(datetime.now().timestamp())
    path = f'{cwd}/models/{name}_{timestamp}.pt'
    torch.save(model.state_dict(), path)
