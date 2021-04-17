import torch
import numpy as np


def state_to_dict(state: torch.Tensor) -> dict:
    # state: tensor(4, 4, 4)
    # find the positions of player, win, pit, wall
    pos = []
    for s in state:
        pos.append(np.array(np.nonzero(s == 1)).flatten().tolist())
    player, win, pit, wall = pos
    return {"player": player, "win": win, "pit": pit, "wall": wall}
