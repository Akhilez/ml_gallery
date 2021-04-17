from omegaconf import DictConfig
from pettingzoo.classic import connect_four_v3
from app.rl.dqn.dqn import train_dqn
from app.rl.dqn.env_wrapper import (
    PettingZooEnvWrapper,
    petting_zoo_random_player,
    NumpyStateMixin,
)
from app.rl.dqn.models import GenericLinearModel
from settings import device


class ConnectXEnvWrapper(PettingZooEnvWrapper, NumpyStateMixin):
    def __init__(self):
        super(ConnectXEnvWrapper, self).__init__(
            env=connect_four_v3.env(), opponent_policy=petting_zoo_random_player
        )


if __name__ == "__main__":

    hp = DictConfig({})

    hp.steps = 20
    hp.batch_size = 2
    hp.max_steps = 10
    hp.lr = 1e-3
    hp.epsilon_exploration = 0.1
    hp.gamma_discount = 0.9

    model = GenericLinearModel(2 * 6 * 7, [10], 7, flatten=True).float().to(device)

    train_dqn(ConnectXEnvWrapper, model, hp, name="Connect4")
