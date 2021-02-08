import numpy as np

from base import GridWorldBase


class GridWorldRandom(GridWorldBase):

    def predict(self, env):
        return {'move': np.random.randint(0, 4, 1)}
