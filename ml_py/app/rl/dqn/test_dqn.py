from unittest import TestCase, mock
from unittest.mock import Mock
import torch
from app.rl.dqn import utils


class TestUtils(TestCase):
    @mock.patch("app.rl.dqn.utils.torch.rand", return_value=1)
    def test_sample_action_exploit(self, *_):
        q_values = torch.tensor([1, 6, 3]).float()
        valid_actions = [0, 2]
        epsilon = 0.1
        expected_action = 2

        sampled_action = utils.sample_action(q_values, valid_actions, epsilon)

        self.assertEqual(sampled_action, expected_action)

    @mock.patch("app.rl.dqn.utils.torch.rand", return_value=0)
    @mock.patch("app.rl.dqn.utils.torch.randint", return_value=0)
    def test_sample_action_explore(self, *_):
        q_values = torch.tensor([1, 6, 3]).float()
        valid_actions = [0, 2]
        epsilon = 0.1
        expected_action = 0

        sampled_action = utils.sample_action(q_values, valid_actions, epsilon)

        self.assertEqual(sampled_action, expected_action)

    def test_reset_envs_that_took_too_long(self):
        envs = [Mock(), Mock(), Mock()]
        steps_before = torch.tensor([3, 2, 1])
        dones_after = torch.tensor([0, 0, 1])
        max_steps = 4

        # envs_to_be_reset = [1, 0, 1]
        steps_after = torch.tensor(
            [0, 3, 0]  # max_steps is 4.  # steps2 + step1 = step3  # env is done
        )

        observed_steps_after = utils.reset_envs_that_took_too_long(
            envs, steps_before, dones_after, max_steps
        )

        self.assertTrue(torch.all(steps_after == observed_steps_after))

        envs[0].reset.assert_called()
        envs[1].reset.assert_not_called()
        envs[2].reset.assert_called()
