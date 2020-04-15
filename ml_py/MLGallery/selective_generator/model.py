import numpy as np
import matplotlib.pyplot as plt
from numpy import load


data_path = "../../data/mnist/"


def save_mnist_raw():
    train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
    test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

    print("loaded. saving")

    from numpy import save
    save(f'{data_path}/train.npy', train_data)
    print("Done saving one.")
    save(f'{data_path}/test.npy', test_data)


data = load(f'{data_path}/test.npy')

print(data.shape)
