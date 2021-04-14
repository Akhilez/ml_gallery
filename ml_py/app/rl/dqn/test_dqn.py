from unittest import TestCase, mock
from unittest.mock import patch, Mock

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

    def test_reset_envs_that_took_too_long(self):

        env1 = Mock()
        env2 = Mock()

        dqn.reset_envs_that_took_too_long(
            envs=[env1, env2], steps=torch.tensor([1, 3]), max_steps=3
        )

        env1.reset.assert_not_called()
        env2.reset.assert_called_once()
