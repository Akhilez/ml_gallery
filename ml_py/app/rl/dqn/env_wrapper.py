from abc import ABC, abstractmethod


class EnvWrapper(ABC):
    def __init__(self, env):
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
    def get_legal_actions(self):
        pass
