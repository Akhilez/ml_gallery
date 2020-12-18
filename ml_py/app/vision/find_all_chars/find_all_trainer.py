import torch
from lib.mnist_aug.mnist_augmenter import DataManager, MNISTAug
from app.vision.find_all_chars.model import MnistDetector


def train(model, x_train, y_train, x_test, y_test):
    pass



def main():
    k = 9
    H = 112
    W = 112
    Wp = 22
    Hp = 22
    b_regions = 256

    threshold_p = 0.6
    threshold_n = 0.3

    dm = DataManager()
    dm.load()

    aug = MNISTAug()
    x_train, y_train = aug.get_augmented(dm.x_train, dm.y_train, 10)
    x_test, y_test = aug.get_augmented(dm.x_test, dm.y_test, 2)

    x_train = torch.tensor(x_train, dtype=torch.float32).view((-1, 1, H, W))
    x_test = torch.tensor(x_test, dtype=torch.float32).view((-1, 1, H, W))

    model = MnistDetector


if __name__ == '__main__':
    main()
