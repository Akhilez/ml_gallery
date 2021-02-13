from typing import List

import numpy as np
import torch
from torch import nn

from base import GridWorldBase
from pg import GWPgModel
from utils import load_model, CWD, device


class GridWorldQ(GridWorldBase):

    def __init__(self):
        self.model = load_model(CWD, GWPgModel(10, [50]).double().to(device), name='q.pt')

    def predict(self, env):
        y = self.model(GWPgModel.convert_inputs([env]))
        action = int(y[0].argmax(0))
        return {'move': action}
