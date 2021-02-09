import numpy as np

from base import GridWorldBase


class GridWorldPG(GridWorldBase):

    def predict(self, env):
        # TODO: Implement
        return {'move': int(np.random.randint(0, 4, 1))}

