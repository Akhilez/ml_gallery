import gym
from griddly import gd

from app.rl.dqn.env_wrapper import GriddlyEnvWrapper, TensorStateMixin


class SokobanEnvWrapper(GriddlyEnvWrapper, TensorStateMixin):
    def __init__(self):
        super().__init__()
        self.env = gym.make(
            "GDY-Sokoban-v0",
            global_observer_type=gd.ObserverType.VECTOR,
            player_observer_type=gd.ObserverType.VECTOR,
        )
