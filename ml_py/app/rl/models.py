from typing import List
from torch import nn


class GenericLinearModel(nn.Module):
    def __init__(
        self, in_size: int, units: List[int], out_size: int, flatten: bool = False
    ):
        super().__init__()

        flatten_layer = [nn.Flatten()] if flatten else []
        self.first = nn.Sequential(
            *flatten_layer, nn.Linear(in_size, units[0]), nn.ReLU(), nn.Dropout(0.3)
        )

        self.hidden = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(units[i], units[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )
                for i in range(len(units) - 1)
            ]
        )

        self.out = nn.Linear(units[-1], out_size)

    def forward(self, x):
        x = self.first(x)
        for hidden in self.hidden:
            x = hidden(x)
        x = x.flatten(1)
        return self.out(x)


class GenericConvModel(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        in_channels: int,
        channels: List[int],
        out_size: int,
    ):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.hidden = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )
                for i in range(len(channels) - 1)
            ]
        )

        self.out = nn.Linear(channels[-1] * height * width, out_size)

    def forward(self, x):
        x = self.first(x)
        for hidden in self.hidden:
            x = hidden(x)
        x = x.flatten(1)
        return self.out(x)
