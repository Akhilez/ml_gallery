from random import shuffle
from typing import Optional, Iterable

import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from collections import deque


class MarioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
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

    def forward(self, x):
        return self.model(x)


class MarioICM(nn.Module):
    def __init__(self):
        super().__init__()

        # (S1, S2) -> a
        self.inverse_module = nn.Sequential(
            nn.Linear(288 * 2, 100),
            nn.ReLU(),
            nn.Linear(100, 12),
        )

        # (S1, a) -> S2
        self.forward_module = nn.Sequential(
            nn.Linear(288 + 12, 1000),
            nn.ReLU(),
            nn.Linear(1000, 288),
        )

        # S1 -> S1~
        self.encode_module = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(288, 512),
            nn.ELU(),
            nn.Linear(512, 288),
        )

    def forward(self, state1, action, state2):

        # 1. Encode the states from encoding module
        state1 = self.encode_module(state1)
        state2 = self.encode_module(state2)

        # 2. Predict action from inverse module
        states = torch.cat((state1, state2), dim=1)
        action_pred = self.inverse_module(states)

        # 3. Predict state2 from forward module
        action_onehot = to_onehot(action, 12)
        state_action = torch.cat((state1.detach(), action_onehot), dim=1)
        state2_pred = self.forward_module(state_action)

        return action_pred, state2, state2_pred


class ExperienceReplay:
    def __init__(self, buffer_size: int = 500, batch_size: int = 100):
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.memory = []
        self.counter = 0

    def add(
        self, state1: torch.Tensor, action: int, reward: float, state2: torch.Tensor
    ):
        self.counter += 1
        if self.counter % 500 == 0:
            shuffle(self.memory)

        memory_tuple = (
            state1,
            action,
            reward,
            state2,
        )
        if len(self.memory) < self.buffer_size:
            self.memory.append(memory_tuple)
        else:
            rand_index = np.random.randint(0, self.buffer_size - 1)
            self.memory[rand_index] = memory_tuple

    def get_batch(self):
        batch_size = (
            len(self.memory) if len(self.memory) < self.batch_size else self.batch_size
        )

        if len(self.memory) < 1:
            print("Error: No data in memory.")
            return None

        indices = np.random.choice(
            np.arange(len(self.memory)), batch_size, replace=False
        )
        batch = [self.memory[i] for i in indices]  # batch is a list of tuples

        state1_batch = torch.stack([x[0].squeeze(dim=0) for x in batch], dim=0)
        action_batch = torch.LongTensor([x[1] for x in batch])
        reward_batch = torch.tensor([x[2] for x in batch])
        state2_batch = torch.stack([x[3].squeeze(dim=0) for x in batch], dim=0)

        return state1_batch, action_batch, reward_batch, state2_batch


def to_onehot(x, num_classes=12):
    b = np.zeros((len(x), num_classes), dtype=np.int32)
    b[np.arange(len(x)), x] = 1
    return torch.tensor(b)


def downscale_obs(obs, new_size=(42, 42), to_gray=True):
    resized = resize(obs, new_size, anti_aliasing=True)
    if to_gray:
        resized = resized.mean(axis=2)
    return resized


def prepare_state(frame: np.ndarray):
    return (
        torch.from_numpy(downscale_obs(frame, new_size=(42, 42), to_gray=True))
        .float()
        .unsqueeze(dim=0)
        .unsqueeze(dim=0)
    )


def prepare_multi_state(state: torch.Tensor, new_frame: np.ndarray):
    """
    state = tensor of 3 frames for batch of size 1. Shape(1, 3, 42, 42)
    new_frame = ndarray of 1 new frame full-size. shape(3, 240, 256)
    """
    new_frame = prepare_state(new_frame)
    state = torch.cat((state[:, 1:].clone(), new_frame), dim=1)
    return state


def prepare_initial_state(frame: np.ndarray):
    """
    state = ndarray of frame 1 full size. shape(3, 240, 256)
    """
    state = prepare_state(frame).repeat((1, 3, 1, 1))
    return state


def sample_action(q_values, epsilon: Optional[float] = None):
    if epsilon is not None:
        if torch.rand(1) < epsilon:
            return torch.randint(low=0, high=len(q_values), size=(1,))
        else:
            return torch.argmax(q_values)
    else:
        q_values = F.softmax(F.normalize(q_values))
        sampled_action = torch.multinomial(q_values, num_samples=1)
        return sampled_action


def main():
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    dqn_model = MarioModel()
    icm_model = MarioICM()

    state = env.reset()

    state = prepare_initial_state(state)

    q_values = dqn_model(state)
    action = int(torch.argmax(q_values[0]))

    state2, reward, done, info = env.step(action)

    state2 = prepare_initial_state(state2)

    action_pred, state2, state2_pred = icm_model(
        state, torch.IntTensor([action]), state2
    )

    action_pred_argmax = F.softmax(action_pred).argmax(dim=1)[0]

    print(action, action_pred_argmax)

    intrinsic_reward = F.kl_div(state2_pred, state2)

    print(reward, intrinsic_reward)


if __name__ == "__main__":
    main()
