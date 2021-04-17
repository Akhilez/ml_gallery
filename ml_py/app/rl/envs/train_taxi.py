import gym
from typing import Iterable
import torch
from app.rl.dqn.dqn import train_dqn
from app.rl.envs.env_wrapper import GymEnvWrapper
from omegaconf import DictConfig
from app.rl.models import GenericLinearModel
from lib.nn_utils import to_onehot


class TaxiV3EnvWrapper(GymEnvWrapper):
    def __init__(self):
        super().__init__()
        self.env = gym.make("Taxi-v3")

    def get_legal_actions(self):
        return list(range(6))

    @staticmethod
    def get_state_batch(envs: Iterable) -> torch.Tensor:
        return to_onehot([env.state for env in envs], 500).float()


if __name__ == "__main__":

    hp = DictConfig({})

    hp.steps = 10000
    hp.batch_size = 500

    hp.max_steps = 200

    hp.lr = 1e-3
    hp.epsilon_exploration = 0.1
    hp.gamma_discount = 0.9

    hp.units = [100]

    model = GenericLinearModel(in_size=500, units=hp.units, out_size=6)

    train_dqn(TaxiV3EnvWrapper, model, hp, name="TaxiV3")
