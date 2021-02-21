import numpy as np
import matplotlib.pyplot as plt
from numpy import load
import torch
import torchvision

data_path = "../../../data/mnist/"

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        data_path,
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    ),
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        data_path,
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    ),
    shuffle=True,
)


x_train = []
x_test = []
y_test = []
y_train = []

for data in test_loader:
    data[1]
    x_test.append()
test = list(test_loader)
print()
