from torch.utils.data import DataLoader

from lib.mnist_aug.loader import MNISTAugDataset
from app.vision.find_all_chars.model import MnistDetector


def train(model, train_set, epochs, batch_size, test_set=None):

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)

    for epoch in range(epochs):
        for i_batch, batch in enumerate(train_loader):
            print(i_batch)

            x = batch['x']
            y = batch['y']

            print(x.shape)
            print(y)
            return


def main():

    train_set = MNISTAugDataset(10)
    test_set = MNISTAugDataset(2)

    model = MnistDetector()

    epochs = 1
    batch_size = 2

    train(model, train_set, epochs, batch_size, test_set=test_set)


if __name__ == '__main__':
    main()
