from unittest import TestCase

import gym_super_mario_bros
import torch
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from app.rl.icm.mario import downscale_obs, prepare_state, to_onehot


class TestMario(TestCase):
    def setUp(self) -> None:
        self.env = gym_super_mario_bros.make("SuperMarioBros-v0")
        self.env = JoypadSpace(self.env, COMPLEX_MOVEMENT)

    def test_downscale_obs(self):
        state = self.env.reset()

        # state.shape = (240, 256, 3)
        self.assertEqual(state.shape, (240, 256, 3))

        # downscale state to (42, 42)
        downscaled_state = downscale_obs(state, new_size=(42, 42), to_gray=True)
        self.assertEqual(downscaled_state.shape, (42, 42))

    def test_prepare_state(self):
        state = self.env.reset()
        # state.shape = (240, 256, 3)

        # Prepared state = shape [1, 1, 42, 42]
        prepared_state = prepare_state(state)
        self.assertEqual(prepared_state.shape, (1, 1, 42, 42))

    def test_onehot(self):
        action = 4
        action_one_hot = [0] * 12
        action_one_hot[action] = 1
        action_one_hot = torch.tensor(action_one_hot)

        onehot_result = to_onehot([action], 12)

        equals = action_one_hot.tolist() == onehot_result[0].tolist()
        self.assertTrue(equals)
