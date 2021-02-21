from abc import ABC, abstractmethod

import numpy as np

from gym_grid_world.envs import GridWorldEnv


class GridWorldBase(ABC):
    @abstractmethod
    def predict(self, env: GridWorldEnv) -> dict:
        pass

    @staticmethod
    def get_item_positions(state):
        pos = []
        for s in state:
            pos.append(np.array(np.nonzero(s == 1)).flatten().tolist())
        return pos


class GridWorldRandom(GridWorldBase):
    def predict(self, env):
        return {"move": int(np.random.randint(0, 4, 1))}
