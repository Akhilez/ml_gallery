from typing import Tuple, Any
import gym_super_mario_bros
import numpy as np
import torch
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from omegaconf import DictConfig
from torch import nn
from app.rl.dqn.dqn import train_dqn
from app.rl.envs.env_wrapper import (
    TensorStateMixin,
    GymEnvWrapper,
    TimeOutLostMixin,
    timeout_lost,
    step_incrementer,
    reset_incrementer,
)
from skimage.transform import resize
from settings import device


class MarioEnvWrapper(GymEnvWrapper, TensorStateMixin):
    max_steps = 10  # TODO: Fix this
    reward_range = (-100, 100)  # TODO: Fix this

    def __init__(self):
        super().__init__()
        self.env = gym_super_mario_bros.make("SuperMarioBros-v0")
        self.env = JoypadSpace(self.env, COMPLEX_MOVEMENT)
        self.history_size = 3
        self.action_repeats = 6

    @timeout_lost
    @step_incrementer
    def step(self, action: int, **kwargs) -> Tuple[Any, Any, bool, dict]:
        for _ in range(self.action_repeats):
            frame, self.reward, self.done, self.info = self.env.step(action)
            self.state = prepare_multi_state(self.state, frame)
            if self.done:
                break
        return self.state, self.reward, self.done, self.info

    @reset_incrementer
    def reset(self):
        frame = self.env.reset()
        self.state = prepare_initial_state(frame, self.history_size)
        self.done = False
        return self.state

    def get_legal_actions(self):
        return list(range(12))


def prepare_initial_state(frame: np.ndarray, history_size: int):
    """
    state = ndarray of frame 1 full size. shape(3, 240, 256)
    """
    state = prepare_state(frame).repeat((history_size, 1, 1))
    return state


def prepare_state(frame: np.ndarray):
    return (
        torch.from_numpy(downscale_obs(frame, new_size=(42, 42), to_gray=True))
        .float()
        .unsqueeze(dim=0)
    )


def downscale_obs(obs, new_size=(42, 42), to_gray=True):
    resized = resize(obs, new_size, anti_aliasing=True)
    if to_gray:
        resized = resized.mean(axis=2)
    return resized


def prepare_multi_state(state: torch.Tensor, new_frame: np.ndarray):
    """
    state = tensor of 3 frames. Shape(3, 42, 42)
    new_frame = ndarray of 1 new frame full-size. shape(3, 240, 256)
    """
    new_frame = prepare_state(new_frame)
    state = torch.cat((state[1:].clone(), new_frame), dim=0)
    return state


model = (
    nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
        nn.ELU(),
        nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
        nn.ELU(),
        nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
        nn.ELU(),
        nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear(288, 100),
        nn.ELU(),
        nn.Linear(100, 12),
    )
    .float()
    .to(device)
)


if __name__ == "__main__":

    hp = DictConfig({})

    hp.steps = 2000
    hp.batch_size = 2
    hp.env_record_freq = 500
    hp.env_record_duration = 100
    hp.max_steps = 1000
    hp.lr = 1e-3
    hp.epsilon_exploration = 0.1
    hp.gamma_discount = 0.9

    train_dqn(MarioEnvWrapper, model, hp, name="Mario")
