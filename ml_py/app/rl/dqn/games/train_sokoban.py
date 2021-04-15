import gym
import numpy as np
from griddly import gd

from app.rl.dqn.env_wrapper import GriddlyEnvWrapper, TensorStateMixin


class SokobanEnvWrapper(GriddlyEnvWrapper, TensorStateMixin):
    def __init__(self):
        super().__init__()
        self.env = gym.make(
            "GDY-Sokoban-v0",
            global_observer_type=gd.ObserverType.VECTOR,
            player_observer_type=gd.ObserverType.VECTOR,
            level=1,
        )


env = SokobanEnvWrapper()
env.reset()
env.render()
while True:
    actions = env.get_legal_actions()
    action = np.random.choice(actions)
    env.step(action)
    env.render()
    if env.is_done():
        env.reset()
