from abc import ABC, abstractmethod
from typing import Optional

from gym import Env


class EnvWrapper(ABC, Env):
    def __init__(self, env=None, *kwargs):
        self.env = env

    @abstractmethod
    def step(self, action, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def is_done(self):
        pass

    @abstractmethod
    def render(self, mode="human"):
        pass

    @abstractmethod
    def get_legal_actions(self):
        pass

    def close(self):
        pass


class GymEnvWrapper(EnvWrapper, ABC):
    def __init__(self, env: Optional[Env] = None):
        super().__init__(env)
        self.state = None
        self.reward = None
        self.done = False
        self.info = None

    def step(self, action, **kwargs):
        self.state, self.reward, self.done, self.info = self.env.step(action)

    def reset(self):
        return self.env.reset()

    def is_done(self):
        return self.done

    def render(self, mode="human"):
        return self.env.render(mode)
