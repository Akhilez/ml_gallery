import os
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

models_path = "./models"

TL = "top-left"
T = "top"
TR = "top-right"
L = "left"
C = "center"
R = "right"
BL = "bottom-left"
B = "bottom"
BR = "bottom-right"

index_to_pos = [TL, T, TR, L, C, R, BL, B, BR]


class PositionalEncoder2D(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        super().__init__()
        self.w = (torch.rand((1, c, h, w)) - 0.5).requires_grad_().to(device)

    def forward(self, x):
        batch_size = x.shape[0]
        encoded = self.w.expand((batch_size, -1, -1, -1))
        return torch.cat((x, encoded), 1)


class PositionalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.pos_enc = PositionalEncoder2D(10, 10, 10)
        self.positional_compression = nn.Sequential(
            nn.Conv2d(32 + 10, 64, 3),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            nn.LeakyReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.LeakyReLU(), nn.Linear(64, 10), nn.Softmax()
        )
        self.position_classifier = nn.Sequential(
            nn.Linear(128, 64), nn.LeakyReLU(), nn.Linear(64, 9), nn.Softmax()
        )

    def forward(self, x):
        features = self.features(x)
        positional_features = self.pos_enc(features)
        compressed = self.positional_compression(positional_features)
        flattened = torch.flatten(compressed, 1)
        cls = self.classifier(flattened)
        pos = self.position_classifier(flattened)
        return cls, pos


class PositionalCNN:
    def __init__(self):
        self.model = self.load_model()

    def predict(self, image):
        self.model.eval()
        with torch.no_grad():

            image = torch.tensor(image).view((1, 1, 112, 112)).to(device)
            cls, pos = self.model(image)

            _, pos = pos.max(1)
            _, cls = cls.max(1)

            pos = index_to_pos[int(pos[0])]

            return int(cls[0]), pos

    def load_model(self, latest=True, name=None):
        model = PositionalClassifier()
        try:
            if latest:
                name = max(os.listdir(models_path))
            model.load_state_dict(
                torch.load(f"{models_path}/{name}", map_location=torch.device(device))
            )
            print(f"Loading model {name}")
        except Exception as e:
            print(e)
        return model
