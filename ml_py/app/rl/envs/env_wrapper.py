from abc import ABC, abstractmethod
from typing import Optional, Iterable, Tuple, Any, Type, List, Callable, Sequence, Union
import itertools
import numpy as np
import torch
from gym import Env
from pettingzoo import AECEnv
from settings import device


class EnvWrapper(ABC, Env):
    reward_range = (-np.inf, np.inf)
    max_steps = np.inf

    def __init__(self, env=None, *kwargs):
        self.env = env
        self.state = None
        self.reward = None
        self.done = False
        self.info = {}

        self.step_count = 0

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


def step_incrementer(f):
    def step_increment(self, action, *args, **kwargs):
        state, reward, done, info = f(self, action, *args, **kwargs)
        self.step_count += 1
        return state, reward, done, info

    return step_increment


def reset_incrementer(f):
    def inner(self, *args, **kwargs):
        returned = f(self, *args, **kwargs)
        self.step_count = 0
        return returned

    return inner


def timeout_lost(f):
    def inner(self, action, *args, **kwargs):
        assert self.step_count < self.max_steps
        state, reward, done, info = f(self, action, *args, **kwargs)
        if self.step_count >= self.max_steps:
            done = True
            self.done = True
            reward = self.reward_range[0]
            info["timed_out"] = True
        return state, reward, done, info

    return inner


class TimeOutLostMixin(EnvWrapper, ABC):
    @timeout_lost
    @step_incrementer
    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)

    @reset_incrementer
    def reset(self):
        return super().reset()

    def is_done(self):
        return self.step_count >= self.max_steps or super().is_done()


class BatchEnvWrapper(EnvWrapper):
    def __init__(self, env_class: Type[EnvWrapper], batch_size: int):
        super(BatchEnvWrapper, self).__init__()
        self.env_class = env_class
        self.batch_size = batch_size
        self.envs: List[EnvWrapper] = [env_class() for _ in range(batch_size)]

    def step(self, actions, **kwargs) -> Tuple[List, List, List[bool], List[dict]]:
        assert len(self.envs) > 0
        assert len(self.envs) == len(actions)

        self.state = []
        self.reward = []
        self.done = []
        self.info = []

        for env, action in zip(self.envs, actions):
            next_state, reward, done, info = env.step(action)

            self.state.append(next_state)
            self.reward.append(reward)
            self.done.append(done)
            self.info.append(info)

            if done:
                env.reset()

        return self.state, self.reward, self.done, self.info

    def reset(self):
        self.state = [env.reset() for env in self.envs]
        self.reward = [None for _ in self.envs]
        self.done = [False for _ in self.envs]
        self.info = [{} for _ in self.envs]
        return self.state

    def is_done(self, reduction: Optional[str] = None) -> Union[bool, Sequence[bool]]:
        self.done = [env.is_done() for env in self.envs]
        if reduction == "all":
            return all(self.done)
        if reduction == "any":
            return any(self.done)
        return self.done

    def render(self, **kwargs):
        return [env.render(**kwargs) for env in self.envs]

    def get_legal_actions(self):
        return [env.get_legal_actions() for env in self.envs]

    def get_state_batch(self) -> torch.Tensor:
        return self.env_class.get_state_batch(self.envs)


class GymEnvWrapper(EnvWrapper, ABC):
    def __init__(self, env: Optional[Env] = None):
        super().__init__(env)
        self.metadata = env.metadata if env else None
        self.state = None
        self.reward = None
        self.done = False
        self.info = {}

    def step(self, action, **kwargs):
        self.state, self.reward, self.done, self.info = self.env.step(action)
        return self.state, self.reward, self.done, self.info

    def reset(self):
        self.state = self.env.reset()
        self.done = False
        self.reward = None
        self.info = {}
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
        self.done = False
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
        self.done = False
        return self.state

    def step(self, action, **kwargs):
        self.env.step(action)

        if not any(self.env.dones.values()):
            self.env.step(self.opponent_policy(self.env))

        self.state = self.env.observe(self.env.agent_selection)["observation"]
        self.done = any(self.env.dones.values())
        self.reward = self.env.rewards[self.learner]
        self.info = self.env.infos[self.learner]

        return self.state, self.reward, self.done, self.info

    def get_legal_actions(self):
        observation = self.env.observe(self.env.agent_selection)
        return np.nonzero(observation["action_mask"])[0]


def petting_zoo_random_player(env: AECEnv) -> int:
    action_mask = env.observe(env.agent_selection)["action_mask"]
    actions = np.nonzero(action_mask)[0]
    return np.random.choice(actions)


class DoneIgnoreBatchedEnvWrapper(BatchEnvWrapper):
    def step(self, actions, **kwargs) -> Tuple[List, List, List[bool], List[dict]]:
        assert len(self.envs) > 0
        assert len(self.envs) == len(actions)

        for index, (env, action) in enumerate(zip(self.envs, actions)):
            if env.is_done():
                continue

            next_state, reward, done, info = env.step(action)

            self.state[index] = next_state
            self.reward[index] = reward
            self.done[index] = done
            self.info[index] = info

        return self.state, self.reward, self.done, self.info
