from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np
import torch


class ActionSampler(ABC):
    @abstractmethod
    def __call__(
        self, valid_actions: Sequence[Sequence[int]], *args, **kwargs
    ) -> Sequence[int]:
        pass


class RandomActionSampler(ActionSampler):
    def __call__(self, valid_actions, **_) -> Sequence[int]:
        return [np.random.choice(actions) for actions in valid_actions]


class EpsilonRandomActionSampler(ActionSampler):
    def __call__(
        self, valid_actions, q_values: torch.Tensor, epsilon: float
    ) -> Sequence[int]:

        return [
            self.sample_action(q_values[i], valid_actions[i], epsilon)
            for i in range(len(q_values))
        ]

    @staticmethod
    def sample_action(q_values, valid_actions, epsilon):
        # q_values of shape (num_actions)
        # valid_actions of shape (any)

        if torch.rand((1,)) < epsilon:
            # explore
            action_index = torch.randint(low=0, high=len(valid_actions), size=(1,))
        else:
            action_index = q_values[valid_actions].argmax(0)

        return valid_actions[action_index]


class ProbabilityActionSampler(ActionSampler):
    def __call__(self, valid_actions, probs, noise: float = 0) -> Sequence[int]:
        """
        valid actions = list of list of int
        probs: tensor of shape (batch, num_actions). Probability distributions
        """
        final_actions = []
        probs = probs + noise
        for i in range(len(valid_actions)):
            actions = valid_actions[i]
            valid_probs = probs[i][actions]
            action = torch.multinomial(valid_probs, 1)
            final_actions.append(int(action))

        return final_actions
