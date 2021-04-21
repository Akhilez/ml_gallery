from heapq import heappush, heappushpop
from typing import Tuple, Optional, Callable, List
import numpy as np
import torch


class PrioritizedReplay:
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        delete_freq: int = None,
        delete_percentage: float = None,
        transform: Optional[Callable[[List[Tuple[float, Tuple]]], Tuple]] = None,
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.delete_freq = delete_freq if delete_freq is not None else 500
        self.delete_percentage = (
            delete_percentage if delete_percentage is not None else 0.5
        )
        self.transform = transform

        self.memory = []
        self.counter = 0

    def add(self, loss: float, data: Tuple):
        self.counter += 1
        if self.counter % self.delete_freq == 0:
            delete_size = int(len(self.memory) * self.delete_percentage)
            self.memory, _ = delete_random(self.memory, delete_size)

        if len(self.memory) < self.buffer_size:
            # Adding counter to break tie with other elements
            heappush(self.memory, (loss, self.counter, data))
        else:
            heappushpop(self.memory, (loss, self.counter, data))

    def add_batch(self, losses, data: Tuple):
        for i in range(len(losses)):
            self.add(float(losses[i]), tuple([data[j][i] for j in range(len(data))]))

    def get_batch(self):
        batch_size = (
            len(self.memory) if len(self.memory) < self.batch_size else self.batch_size
        )

        if len(self.memory) < 1:
            if self.transform:
                return self.transform([])
            return torch.rand(0)

        self.memory, batch = delete_random(self.memory, batch_size)

        if self.transform:
            batch = self.transform(batch)

        return batch


def delete_random(array, size):
    indices = np.random.choice(np.arange(len(array)), size, replace=False)
    deleted = [array[i] for i in indices]
    for i in sorted(indices, reverse=True):
        del array[i]
    return array, deleted


def state_action_reward_state_2_transform(
    data: [List[Tuple[float, int, Tuple]]]
) -> Tuple:
    if len(data) == 0:
        empty = torch.rand(0)
        return empty, empty, empty, empty
    state1_batch = torch.stack([x[2][0] for x in data])
    action_batch = torch.LongTensor([x[2][1] for x in data])
    reward_batch = torch.tensor([x[2][2] for x in data])
    state2_batch = torch.stack([x[2][3] for x in data])
    return state1_batch, action_batch, reward_batch, state2_batch
