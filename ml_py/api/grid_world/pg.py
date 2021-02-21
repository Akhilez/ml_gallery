from typing import List

import numpy as np
import torch
from torch import nn

from base import GridWorldBase
from utils import load_model, CWD, device


class GWPgModel(nn.Module):
    def __init__(self, size: int, units: List[int]):
        super().__init__()
        self.size = size

        self.first = nn.Sequential(
            nn.Conv2d(4, units[0], kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.3)
        )

        self.hidden = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(units[i], units[i + 1], kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )
                for i in range(len(units) - 1)
            ]
        )

        self.out = nn.Linear(self.size * self.size * units[-1], 4)

    def forward(self, x):
        x = self.first(x)
        for hidden in self.hidden:
            x = hidden(x)
        x = x.flatten(1)
        return self.out(x)

    @staticmethod
    def convert_inputs(envs):
        inputs = np.array([env.state for env in envs])
        return torch.tensor(inputs).double().to(device)


class GridWorldPG(GridWorldBase):
    def __init__(self):
        self.model = load_model(
            CWD, GWPgModel(4, [25, 25]).double().to(device), name="pg.pt"
        )

    def predict(self, env):
        y = self.model(GWPgModel.convert_inputs([env]))
        action = int(y[0].argmax(0))
        return {"move": action}
