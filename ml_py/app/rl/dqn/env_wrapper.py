from abc import ABC, abstractmethod
from typing import Optional, Iterable, Tuple, Any, Type, List
import itertools
import torch
from gym import Env

from settings import device


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


class TensorStateMixin(EnvWrapper, ABC):
    @staticmethod
    def get_state_batch(envs: Iterable) -> torch.Tensor:
        return torch.tensor([env.state for env in envs]).float().to(device)


class BatchEnvWrapper(EnvWrapper):
    def __init__(self, env_class: Type[EnvWrapper], batch_size: int):
        super(BatchEnvWrapper, self).__init__()
        self.env_class = env_class
        self.batch_size = batch_size
        self.envs: List[EnvWrapper] = [env_class() for _ in range(batch_size)]

    def step(self, actions, **kwargs) -> Tuple[List, List, List[bool], List[dict]]:
        assert len(self.envs) > 0
        assert len(self.envs) == len(actions)

        states = []
        rewards = []
        dones = []
        infos = []

        for env, action in zip(self.envs, actions):
            next_state, reward, done, info = env.step(action)

            states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

            if done:
                env.reset()

        return states, rewards, dones, infos

    def reset(self):
        return [env.reset() for env in self.envs]

    def is_done(self) -> List[bool]:
        return [env.is_done() for env in self.envs]

    def render(self, **kwargs):
        return [env.render(**kwargs) for env in self.envs]

    def get_legal_actions(self):
        return [env.get_legal_actions() for env in self.envs]

    def get_state_batch(self) -> torch.Tensor:
        return self.env_class.get_state_batch(self.envs)


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
