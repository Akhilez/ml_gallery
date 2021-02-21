import torch
import os

grid_size = 4
mode = "random"

device = "cuda" if torch.cuda.is_available() else "cpu"
CWD = os.path.dirname(os.path.abspath(__file__))


class AlgorithmTypes:
    random = "random"
    pg = "pg"
    q = "q"
    mcts = "mcts"
    alpha_zero = "alphaZero"
    mu_zero = "mu_zero"


def load_model(
    cwd: str, model: torch.nn.Module, name: str = None, latest: bool = False
):
    models_path = f"{cwd}/models"

    if latest:
        name = max(os.listdir(models_path))

    print(f"Loading model {name}")
    model.load_state_dict(torch.load(f"{models_path}/{name}"))
    return model
