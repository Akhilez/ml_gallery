import numpy as np
import matplotlib.pyplot as plt
import torch
from random import random
from skimage.transform import resize


class MNISTAug:
    def __init__(self):
        self.dm = DataManager()
        self.scale = 4
        self.overflow = 0.5
        self.min_numbers_out = 4
        self.max_numbers_out = 10

    def get_augmented(self, x: np.ndarray, y: np.ndarray, n_out: int):
        """

        Parameters
        ----------
        x: a tensor of shape [1000, 28, 28]
        y: a tensor of shape [1000, 1]
        n_out: number of output images

        Returns
        -------
        aug_x: a tensor of shape [1000, 112, 112]
        aug_y: a tensor of shape [n_out, numbers_out, 5] | 5 => [class, x1, y1, x2, y2]

        """

        self.dm.load_test()

        x_in = x.shape[1]
        x_out = x.shape[1] * self.scale

        aug_x = np.zeros((x.shape[0], x_out, x_out))

        i = 0

        rand_x = int(random() * x_in * (self.scale - self.overflow))
        rand_y = int(random() * x_in * (self.scale - self.overflow))

        localized_dim_x = min(aug_x - rand_x, x_in)
        localized_dim_y = min(aug_x - rand_y, x_in)

        localized_xi = x[i][:localized_dim_x, :localized_dim_y]
        aug_x[i][rand_x:rand_x + localized_dim_x, rand_y:rand_y + localized_dim_y] += localized_xi

        return aug_x, aug_y


class DataManager:
    def __init__(self):
        from ml_py.settings import BASE_DIR
        self.dir = f'{BASE_DIR}/data/mnist/numbers'

        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

    def load(self):
        self.load_train()
        self.load_test()

    def load_train(self):
        self.x_train = np.load(f'{self.dir}/x_train.npy')
        self.y_train = np.load(f'{self.dir}/y_train.npy')

    def load_test(self):
        self.x_test = np.load(f'{self.dir}/x_test.npy')
        self.y_test = np.load(f'{self.dir}/y_test.npy')

    @staticmethod
    def plot_num(x):
        plt.imshow(x, cmap='gray')
