from typing import List
import torch
from app.rl.envs.env_wrapper import EnvWrapper


def sample_actions(
    q_values: torch.Tensor, valid_actions: List[List[int]], epsilon: float
) -> torch.IntTensor:
    # q_values: tensor of shape (batch, num_actions)
    # Valid_actions: tensor of shape (batch, any)
    return torch.IntTensor(
        [
            sample_action(q_values[i], valid_actions[i], epsilon)
            for i in range(len(q_values))
        ]
    )


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
    envs: List[EnvWrapper], steps: torch.Tensor, dones: torch.Tensor, max_steps: int
):
    """
    Pseudo code:

    steps = [3, 2, 5]
    dones = [0, 0, 1]

    not_dones = [1, 1, 0]
    ones = [1, 1, 1]

    desired_result = [4, 3, 0]

    steps + ones = [4, 3, 6]
    steps_new = (steps + ones) * not_dones = [4, 3, 0]

    max_steps = 4

    maxed_out = steps_new >= max_steps = [1, 0, 0]
    steps_new[maxed_out] = 0

    maxed_indices = nonzero(maxed_out)
    envs[maxed_indices].reset()

    """

    not_dones = torch.logical_not(dones)
    steps = (steps + torch.ones(len(envs))) * not_dones

    maxed_out = steps >= max_steps
    steps[maxed_out] = 0

    reset_indices = torch.nonzero(steps == 0).flatten(0)
    for index in reset_indices:
        envs[index].reset()

    return steps
