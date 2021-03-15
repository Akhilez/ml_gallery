from heapq import heappush, heappop, heapify, heappushpop
from typing import Tuple, Any, List
import numpy as np

"""

Life Cycle:

- create a queue
- Add experiences to the queue (for each batch)
    - Add an experience only if loss > existing loss
    - If size > desired, keep popping
- sample random experiences (for each batch)
    - if size == max, remove them from the queue

"""


class PrioritizedReplay:
    def __init__(self, max_size: int):
        self.queue = []
        self.max_size = max_size

    def put(self, experiences: List[Tuple[float, Any]]):
        for experience in experiences:
            try:
                if len(self.queue) > self.max_size:
                    heappushpop(self.queue, experience)
                else:
                    heappush(self.queue, experience)
            except Exception as e:
                print(e)
                # TODO: Oh man, do not put duplicates!

    def sample(self, num_samples: int) -> List[Any]:
        if len(self.queue) == 0:
            return []
        if num_samples >= len(self.queue):
            return self.queue

        random_indices = np.random.choice(
            range(len(self.queue)), num_samples, replace=False
        )
        random_samples = []

        for index in random_indices:
            random_samples.append(self.queue[index])

        if len(self.queue) == self.max_size:
            # Remove if q size == max
            for i in sorted(random_indices, reverse=True):
                del self.queue[i]

        return random_samples
