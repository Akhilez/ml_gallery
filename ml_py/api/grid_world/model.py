import numpy as np


grid_size = 10
mode = 'random'


class GridWorld:
    @staticmethod
    def get_item_positions(state):
        pos = []
        for s in state:
            pos.append(np.array(np.nonzero(s == 1)).flatten().tolist())
        return pos

