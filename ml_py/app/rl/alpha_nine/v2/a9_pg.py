from typing import List
import torch
from torch import nn
from gym_nine_mens_morris.envs.nmm_v2 import NineMensMorrisEnvV2

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class A9PgModel(nn.Module):
    """
    1. 24 * 3 = 72
    2. ...
    3. 24: pos1, 24: pos2, 24 left, 24 up, 24 right, 24 left, 24 kill = 7 * 24 = 168
    """
    def __init__(self, units: List[int]):
        super().__init__()

        self.first = nn.Sequential(
            nn.Linear(24 * 3, units[0]),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.hidden = nn.ModuleList([nn.Sequential(
            nn.Linear(units[i], units[i + 1]),
            nn.ReLU(),
            nn.Dropout(0.3)
        ) for i in range(len(units) - 1)])

        self.out = nn.Linear(units[-1], 7 * 24)

    def forward(self, x):
        x = self.first(x)
        for hidden in self.hidden:
            x = hidden(x)
        return self.out(x).view((7, 24))

    @staticmethod
    def convert_inputs(states):
        # state: shape(n, 24), mens
        # output: shape(n, 72)
        inputs = []
        for state in states:
            xb = state[0]
            inputs_b = []
            for xi in xb:
                if xi == 1:
                    inputs_b.extend([1, 0, 0])
                elif xi == -1:
                    inputs_b.extend([0, 0, 1])
                else:
                    inputs_b.extend([0, 1, 0])
            inputs.append(inputs_b)
        return torch.tensor(inputs).double().to(device)

    @staticmethod
    def convert_outputs(op):
        # op: shape(n, 7, 24)
        # output: tuple((n, 24), (n, 24), (n, 4, 24), (n, 24))
        pos1 = op[:, 0]
        pos2 = op[:, 1]
        moves = op[:, 2: 6]
        kill = op[:, -1]

        return pos1, pos2, moves, kill


lrs = [1e-3]

epochs = 1
lr = lrs[0]

env = NineMensMorrisEnvV2()
env.reset()

model = A9PgModel([200]).double().to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)

x = model.convert_inputs([env.state])
yh = model(x)

print(yh.shape)

