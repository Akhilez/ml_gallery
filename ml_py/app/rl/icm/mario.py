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


env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, COMPLEX_MOVEMENT)


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
