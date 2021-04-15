from typing import Iterable, List
import torch
from torch import nn

from app.rl.dqn.dqn import train_dqn
from app.rl.dqn.env_wrapper import GymEnvWrapper, TensorStateMixin
from gym_grid_world.envs import GridWorldEnv
from omegaconf import DictConfig
from utils import device


class GridWorldEnvWrapper(GymEnvWrapper, TensorStateMixin):
    def __init__(self):
        super().__init__()
        self.env = GridWorldEnv(size=4, mode="static")

    def get_legal_actions(self):
        return self.env.get_legal_actions()


class GWPgModel(nn.Module):
    def __init__(self, size: int, units: List[int]):
        super().__init__()
        self.size = size

        self.first = nn.Sequential(
            nn.Conv2d(4, units[0], kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.3)
        )

        self.hidden = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(units[i], units[i + 1], kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )
                for i in range(len(units) - 1)
            ]
        )

        self.out = nn.Linear(self.size * self.size * units[-1], 4)

    def forward(self, x):
        x = self.first(x)
        for hidden in self.hidden:
            x = hidden(x)
        x = x.flatten(1)
        return self.out(x)


if __name__ == "__main__":

    hp = DictConfig({})

    hp.steps = 1000
    hp.batch_size = 1000
    hp.env_record_freq = 1000
    hp.env_record_duration = 25

    hp.max_steps = 50
    hp.grid_size = 4

    hp.lr = 1e-3
    hp.epsilon_exploration = 0.1
    hp.gamma_discount = 0.9

    model = GWPgModel(size=hp.grid_size, units=[10]).float().to(device)

    train_dqn(GridWorldEnvWrapper, model, hp, name="SimpleGridWorld")
