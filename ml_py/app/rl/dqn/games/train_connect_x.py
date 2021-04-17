import numpy as np
from pettingzoo.classic import connect_four_v3
from app.rl.dqn.env_wrapper import (
    PettingZooEnvWrapper,
    petting_zoo_random_player,
    NumpyStateMixin,
)


class ConnectXEnvWrapper(PettingZooEnvWrapper, NumpyStateMixin):
    def __init__(self):
        super(ConnectXEnvWrapper, self).__init__(
            env=connect_four_v3.env(), opponent_policy=petting_zoo_random_player
        )


if __name__ == "__main__":
    env = ConnectXEnvWrapper()
    env.reset()
    env.render()
    while True:
        env.step(np.random.choice(env.get_legal_actions()))
        env.render()
        if env.done:
            env.reset()
