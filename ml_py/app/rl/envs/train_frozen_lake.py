from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from typing import Iterable
import torch
from app.rl.dqn.dqn import train_dqn
from app.rl.envs.env_wrapper import GymEnvWrapper
from omegaconf import DictConfig
from app.rl.models import GenericLinearModel
from lib.nn_utils import to_onehot
from settings import device


class FrozenLakeEnvWrapper(GymEnvWrapper):
    max_steps = 50
    reward_range = (-10, 10)

    def __init__(self):
        super().__init__()
        self.env = FrozenLakeEnv(map_name="4x4", is_slippery=True)

    def get_legal_actions(self):
        return list(range(4))

    @staticmethod
    def get_state_batch(envs: Iterable) -> torch.Tensor:
        return to_onehot([env.state for env in envs], 16).float()


if __name__ == "__main__":

    hp = DictConfig({})

    hp.steps = 5000
    hp.batch_size = 500

    hp.max_steps = 200

    hp.lr = 1e-3
    hp.epsilon_exploration = 0.1
    hp.gamma_discount = 0.9

    hp.units = [10]

    model = GenericLinearModel(16, hp.units, 4).double().to(device)

    train_dqn(FrozenLakeEnvWrapper, model, hp, name="FrozenLake")
