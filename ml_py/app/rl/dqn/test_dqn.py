from unittest import TestCase, mock

import torch

from app.rl.dqn import dqn


class TestDqn(TestCase):
    @mock.patch("app.rl.dqn.dqn.torch.rand", return_value=1)
    def test_sample_action_exploit(self, *_):
        q_values = torch.tensor([1, 6, 3]).float()
        valid_actions = [0, 2]
        epsilon = 0.1
        expected_action = 2

        sampled_action = dqn.sample_action(q_values, valid_actions, epsilon)

        self.assertEqual(sampled_action, expected_action)

    @mock.patch("app.rl.dqn.dqn.torch.rand", return_value=0)
    @mock.patch("app.rl.dqn.dqn.torch.randint", return_value=0)
    def test_sample_action_explore(self, *_):
        q_values = torch.tensor([1, 6, 3]).float()
        valid_actions = [0, 2]
        epsilon = 0.1
        expected_action = 0

        sampled_action = dqn.sample_action(q_values, valid_actions, epsilon)

        self.assertEqual(sampled_action, expected_action)
