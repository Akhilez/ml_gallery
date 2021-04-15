from typing import List

import gym
import numpy as np
from griddly import gd
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from torch import nn

from app.rl.dqn.dqn import train_dqn
from app.rl.dqn.env_wrapper import GriddlyEnvWrapper, TensorStateMixin
from settings import device


class SokobanV2L0EnvWrapper(GriddlyEnvWrapper, TensorStateMixin):
    def __init__(self):
        super().__init__()
        self.env = gym.make(
            "GDY-Sokoban---2-v0",
            global_observer_type=gd.ObserverType.VECTOR,
            player_observer_type=gd.ObserverType.VECTOR,
            level=0,
        )


class SokobanFlatModel(nn.Module):
    def __init__(
        self, in_size: int, units: List[int], out_size: int, flatten: bool = False
    ):
        super().__init__()

        flatten_layer = [nn.Flatten()] if flatten else []
        self.first = nn.Sequential(
            *flatten_layer, nn.Linear(in_size, units[0]), nn.ReLU(), nn.Dropout(0.3)
        )

        self.hidden = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(units[i], units[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )
                for i in range(len(units) - 1)
            ]
        )

        self.out = nn.Linear(units[-1], out_size)

    def forward(self, x):
        x = self.first(x)
        for hidden in self.hidden:
            x = hidden(x)
        x = x.flatten(1)
        return self.out(x)


if __name__ == "__main__":

    hp = DictConfig({})

    hp.steps = 10000
    hp.batch_size = 1000
    hp.env_record_freq = 500
    hp.env_record_duration = 50
    hp.max_steps = 200
    hp.lr = 1e-3
    hp.epsilon_exploration = 0.1
    hp.gamma_discount = 0.9

    model = SokobanFlatModel(5 * 7 * 8, [10], 5, flatten=True).float().to(device)

    train_dqn(SokobanV2L0EnvWrapper, model, hp, name="SokobanV2L0")
