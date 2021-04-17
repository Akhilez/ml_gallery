from typing import List
import torch
from app.rl.envs.env_wrapper import EnvWrapper


def sample_actions(
    q_values: torch.Tensor, valid_actions: List[List[int]], epsilon: float
) -> List[int]:
    # q_values: tensor of shape (batch, num_actions)
    # Valid_actions: tensor of shape (batch, any)
    return [
        sample_action(q_values[i], valid_actions[i], epsilon)
        for i in range(len(q_values))
    ]


def sample_action(
    q_values: torch.Tensor, valid_actions: List[int], epsilon: float
) -> int:
    # q_values of shape (num_actions)
    # valid_actions of shape (any)

    if torch.rand((1,)) < epsilon:
        # explore
        action_index = torch.randint(low=0, high=len(valid_actions), size=(1,))
    else:
        action_index = q_values[valid_actions].argmax(0)

    return valid_actions[action_index]


def reset_envs_that_took_too_long(
    envs: List[EnvWrapper], steps: torch.Tensor, max_steps: int
):
    env_indices = torch.nonzero(steps >= max_steps)
    for index in env_indices:
        envs[index].reset()
