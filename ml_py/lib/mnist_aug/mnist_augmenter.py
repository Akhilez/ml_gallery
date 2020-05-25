import numpy as np


class MNISTAug:
    def __init__(self):
        self.dm = DataManager()


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
