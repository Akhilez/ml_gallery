from abc import ABC, abstractmethod
from typing import Optional, Iterable, Tuple, Any, Type, List, Callable
import itertools
import numpy as np
import torch
from gym import Env
from pettingzoo import AECEnv
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


class NumpyStateMixin(EnvWrapper, ABC):
    @staticmethod
    def get_state_batch(envs: Iterable) -> torch.Tensor:
        return torch.tensor([env.state for env in envs]).float().to(device)


class TensorStateMixin(EnvWrapper, ABC):
    @staticmethod
    def get_state_batch(envs: Iterable) -> torch.Tensor:
        return torch.stack([env.state for env in envs]).float().to(device)


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
        self.metadata = env.metadata
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


class PettingZooEnvWrapper(GymEnvWrapper, ABC):
    def __init__(
        self,
        env: AECEnv,
        opponent_policy: Callable[[AECEnv], int],
        randomize_first: bool = True,
        is_learner_first: bool = False,
    ):
        super(PettingZooEnvWrapper, self).__init__(env)
        self.randomize_first = randomize_first
        self.is_learner_first = is_learner_first
        self.opponent_policy = opponent_policy
        self.learner = None
        self.opponent = None
        self.state = None
        self.reward = None
        self.done = None
        self.info = {}

    def reset(self):
        self.env.reset()
        is_opponent_first = not self.is_learner_first or (
            self.randomize_first and np.random.random() > 0.5
        )
        self.opponent, self.learner = (
            self.env.agents if is_opponent_first else reversed(self.env.agents)
        )
        if is_opponent_first:
            self.env.step(self.opponent_policy(self.env))
        self.state = self.env.observe(self.env.agent_selection)["observation"]
        return self.state

    def step(self, action, **kwargs):
        self.env.step(action)

        if not any(self.env.dones.values()):
            self.env.step(self.opponent_policy(self.env))

        self.state = self.env.observe(self.env.agent_selection)["observation"]
        self.done = any(self.env.dones.values())
        self.reward = self.env.rewards[self.learner]
        self.info = self.env.infos[self.learner]

        return self.state, self.done, self.reward, self.info

    def get_legal_actions(self):
        observation = self.env.observe(self.env.agent_selection)
        return np.nonzero(observation["action_mask"])[0]


def petting_zoo_random_player(env: AECEnv) -> int:
    action_mask = env.observe(env.agent_selection)["action_mask"]
    actions = np.nonzero(action_mask)[0]
    return np.random.choice(actions)
