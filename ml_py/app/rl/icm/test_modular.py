from unittest import TestCase

from app.rl.icm.modular_test import ProcessModule, Compose, DataInit, Loop


class AppenderModule(ProcessModule):
    required_keys = ["array"]

    def run(self):
        self.array.append(len(self.array))


class EpochsLoop(Loop):
    required_keys = [{"hp": ["max_length"]}]

    def terminate(self):
        return len(self.array) >= self.hp.max_length


class TestModular(TestCase):
    def test_data_passage(self):
        graph = Compose(
            DataInit({"array": []}),
            AppenderModule(),
            AppenderModule(),
            AppenderModule(),
        )()

        self.assertEqual(graph.array, [0, 1, 2])

    def test_loop(self):
        graph = Compose(
            DataInit({"array": []}),
            AppenderModule(),
            DataInit({"hp": {"max_length": 10}}),
            EpochsLoop(AppenderModule()),
        )()

        self.assertEqual(graph.array, list(range(10)))
