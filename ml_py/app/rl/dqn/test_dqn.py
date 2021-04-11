from unittest import TestCase, mock

import torch

from app.rl.dqn import dqn


class TestDqn(TestCase):
    @mock.patch(
        "app.rl.dqn.dqn.torch.multinomial", return_value=torch.tensor([1, 1, 1, 1, 0])
    )
    @mock.patch(
        "app.rl.dqn.dqn.torch.randint", return_value=torch.tensor([0, 0, 0, 0, 0])
    )
    def test_sample_action(self, *_):
        q_values = torch.tensor(
            [
                [1, 2, 1, 1],
                [1, 1, 2, 1],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [1, 2, 3, 4],
            ]
        )
        actions = dqn.sample_actions(q_values, 0.1)
        actions_expected = torch.tensor([1, 2, 3, 0, 0])

        self.assertTrue(torch.equal(actions, actions_expected))
