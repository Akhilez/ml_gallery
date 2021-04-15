from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from typing import Iterable, List
import torch
from torch import nn
from app.rl.dqn.dqn import train_dqn
from app.rl.dqn.env_wrapper import GymEnvWrapper
from omegaconf import DictConfig
from lib.nn_utils import to_onehot


class FrozenLakeEnvWrapper(GymEnvWrapper):
    def __init__(self):
        super().__init__()
        self.env = FrozenLakeEnv(map_name="4x4", is_slippery=True)

    def get_legal_actions(self):
        return list(range(4))

    @staticmethod
    def get_state_batch(envs: Iterable) -> torch.Tensor:
        return to_onehot([env.state for env in envs], 16).float()


class FrozenLakeDqnModel(nn.Module):
    def __init__(self, units: List[int]):
        super().__init__()

        self.first = nn.Sequential(nn.Linear(16, units[0]), nn.ReLU(), nn.Dropout(0.3))

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

        self.out = nn.Linear(units[-1], 4)

    def forward(self, x):
        x = self.first(x)
        for hidden in self.hidden:
            x = hidden(x)
        x = x.flatten(1)
        return self.out(x)


if __name__ == "__main__":

    hp = DictConfig({})

    hp.steps = 5000
    hp.batch_size = 500

    hp.max_steps = 200

    hp.lr = 1e-3
    hp.epsilon_exploration = 0.1
    hp.gamma_discount = 0.9

    hp.units = [10]

    model = FrozenLakeDqnModel(units=hp.units)

    train_dqn(FrozenLakeEnvWrapper, model, hp, name="FrozenLake")
