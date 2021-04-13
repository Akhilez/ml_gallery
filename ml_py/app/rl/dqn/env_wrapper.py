from abc import ABC, abstractmethod
from typing import Optional, Iterable, Tuple, Any
import itertools
import torch
from gym import Env


class EnvWrapper(ABC, Env):
    def __init__(self, env=None, *kwargs):
        self.env = env
        self.state = None
        self.reward = None
        self.done = False
        self.info = {}

    @abstractmethod
    def step(self, action, **kwargs) -> Tuple[Any, Any, bool, dict]:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def is_done(self) -> bool:
        pass

    @abstractmethod
    def render(self, mode="human"):
        pass

    @abstractmethod
    def get_legal_actions(self):
        pass

    def close(self):
        pass

    @staticmethod
    @abstractmethod
    def get_state_batch(envs: Iterable) -> torch.Tensor:
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
        return self.state, self.reward, self.done, self.info

    def reset(self):
        self.state = self.env.reset()
        return self.state

    def is_done(self):
        return self.done

    def render(self, mode="human"):
        return self.env.render(mode)


class GriddlyEnvWrapper(GymEnvWrapper, ABC):
    def get_legal_actions(self):
        available_actions = self.env.game.get_available_actions(1)
        location = list(available_actions.keys())[0]
        action = list(available_actions[location])
        valid_actions = self.env.game.get_available_action_ids(location, action)
        valid_actions = list(itertools.chain(*valid_actions.values()))
        return valid_actions
