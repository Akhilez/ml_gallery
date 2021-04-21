from typing import List
from torch import nn
from app.rl.dqn.dqn import train_dqn
from app.rl.dqn.dqn_per import train_dqn_per
from app.rl.envs.env_wrapper import GymEnvWrapper, NumpyStateMixin, TimeOutLostMixin
from gym_grid_world.envs import GridWorldEnv
from omegaconf import DictConfig
from utils import device


class GridWorldEnvWrapper(TimeOutLostMixin, NumpyStateMixin, GymEnvWrapper):
    reward_range = (-10, 10)
    max_steps = 50

    def __init__(self):
        super().__init__(GridWorldEnv(size=4, mode="random"))

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


def dqn_gridworld():

    hp = DictConfig({})

    hp.steps = 1000
    hp.batch_size = 600
    hp.env_record_freq = 100
    hp.env_record_duration = 25

    hp.max_steps = 50
    hp.grid_size = 4

    hp.lr = 1e-3
    hp.epsilon_exploration = 0.1
    hp.gamma_discount = 0.9

    model = GWPgModel(size=hp.grid_size, units=[50]).float().to(device)

    train_dqn(
        GridWorldEnvWrapper, model, hp, project_name="SimpleGridWorld", run_name="dqn"
    )


def dqn_per_gridworld():
    hp = DictConfig({})

    hp.steps = 1000
    hp.batch_size = 500
    hp.replay_batch = 100
    hp.replay_size = 1000
    hp.delete_freq = 100 * (hp.batch_size + hp.replay_size)  # every 100 steps

    hp.env_record_freq = 100
    hp.env_record_duration = 25

    hp.max_steps = 50
    hp.grid_size = 4

    hp.lr = 1e-3
    hp.epsilon_exploration = 0.1
    hp.gamma_discount = 0.9

    model = GWPgModel(size=hp.grid_size, units=[50]).float().to(device)

    train_dqn_per(
        GridWorldEnvWrapper,
        model,
        hp,
        project_name="SimpleGridWorld",
        run_name="dqn_per",
    )


if __name__ == "__main__":
    dqn_per_gridworld()
