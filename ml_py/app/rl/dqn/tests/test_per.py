from unittest import TestCase

import numpy as np

from app.rl.prioritized_replay import PrioritizedReplay


class TestPrioritizedReplay(TestCase):
    def test_adding(self):
        buffer = PrioritizedReplay(5, 2)
        buffer.add(0.1, (1, 1))
        buffer.add_batch([0.2, 0.3], ([6, 7], [3, 4]))

        # expectation = [(0.1, (1, 1)), (0.2, (6, 3)), (0.3, (7, 4))]

        self.assertIn((0.1, 1, (1, 1)), buffer.memory)
        self.assertIn((0.2, 2, (6, 3)), buffer.memory)
        self.assertIn((0.3, 3, (7, 4)), buffer.memory)

        buffer.add_batch([0.4, 0.5, 0.6], ([8, 9, 10], [5, 6, 7]))

        self.assertNotIn((0.1, 1, (1, 1)), buffer.memory)
        self.assertEqual(len(buffer.memory), 5)

        self.assertIn((0.2, 2, (6, 3)), buffer.memory)
        buffer.add(0.7, (11, 8))
        self.assertNotIn((0.2, 2, (6, 3)), buffer.memory)

    def test_add_batch_more_than_limit(self):
        buffer = PrioritizedReplay(2, 1)
        buffer.add_batch([0.1, 0.2, 0.3], ([5, 6, 7], [2, 3, 4]))

        self.assertNotIn((0.1, 1, (5, 2)), buffer.memory)
        self.assertEqual(len(buffer.memory), 2)

    def test_add_duplicate_losses(self):
        buffer = PrioritizedReplay(20, 1)
        dummy = np.ones(10)
        buffer.add_batch(np.zeros(10), (dummy, dummy, dummy, dummy))

        self.assertEqual(len(buffer.memory), 10)

    def test_get_batch(self):
        buffer = PrioritizedReplay(3, 2)
        buffer.add(0.1, (1, 1))
        buffer.add(0.2, (2, 2))
        buffer.add(0.3, (3, 3))

        batch = buffer.get_batch()  # [(0.1, (1, 1)), (0.2, (2, 2))]

        self.assertEqual(len(batch), 2)
        self.assertEqual(len(buffer.memory), 1)

        self.assertNotIn(buffer.memory[0], batch)

        # Dynamically reduce batch size
        batch = buffer.get_batch()
        self.assertEqual(len(batch), 1)
        self.assertEqual(len(buffer.memory), 0)

        # What happens if get batch from empty batch?
        batch = buffer.get_batch()
        self.assertEqual(len(batch), 0)

    def test_duplicate_loss_key(self):
        buffer = PrioritizedReplay(3, 2)
        buffer.add(0.1, (1, 1))
        buffer.add(0.1, (2, 2))
        buffer.add(0.3, (3, 3))
        buffer.add(0.1, (3, 3))
