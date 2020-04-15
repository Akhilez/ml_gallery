import numpy as np
import matplotlib.pyplot as plt
from numpy import load


data_path = "../../data/mnist/"

data = load(f'{data_path}/x_test.npy')

print(data.shape)
