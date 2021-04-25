from omegaconf import DictConfig
from pettingzoo.classic import connect_four_v3
from app.rl.dqn.dqn import train_dqn
from app.rl.envs.env_wrapper import (
    PettingZooEnvWrapper,
    petting_zoo_random_player,
    NumpyStateMixin,
)
from app.rl.models import GenericLinearModel
from settings import device


class ConnectXEnvWrapper(PettingZooEnvWrapper, NumpyStateMixin):
    max_steps = 42  # TODO: Fix this
    reward_range = (-10, 10)  # TODO: Fix this

    def __init__(self):
        super(ConnectXEnvWrapper, self).__init__(
            env=connect_four_v3.env(), opponent_policy=petting_zoo_random_player
        )


def train_dqn_connect4():

    hp = DictConfig({})

    hp.steps = 20
    hp.batch_size = 2
    hp.max_steps = 10
    hp.lr = 1e-3
    hp.epsilon_exploration = 0.1
    hp.gamma_discount = 0.9

    model = GenericLinearModel(2 * 6 * 7, [10], 7, flatten=True).float().to(device)

    train_dqn(ConnectXEnvWrapper, model, hp, name="Connect4")


if __name__ == "__main__":
    train_dqn_connect4()
