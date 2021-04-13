import os
from datetime import datetime
import numpy as np
import torch


def get_scaled_random_weights(shape, min_=-0.5, max_=0.5):
    return torch.FloatTensor(*shape).uniform_(min_, max_).requires_grad_()


def save_model(model: torch.nn.Module, cwd: str, name: str) -> None:
    timestamp = int(datetime.now().timestamp())
    path = f"{cwd}/models/{name}_{timestamp}.pt"
    torch.save(model.state_dict(), path)


def load_model(cwd: str, model: torch.nn.Module, name: str = None, latest: bool = True):
    models_path = f"{cwd}/models"

    if latest:
        name = max(os.listdir(models_path))

    print(f"Loading model {name}")
    model.load_state_dict(torch.load(f"{models_path}/{name}"))
    return model


def to_onehot(x, num_classes: int):
    b = np.zeros((len(x), num_classes), dtype=np.int32)
    b[np.arange(len(x)), x] = 1
    return torch.tensor(b)
