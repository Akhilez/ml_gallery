from torch.utils.data import DataLoader
import torch

from lib.mnist_aug.loader import MNISTAugDataset
from app.vision.find_all_chars.model import MnistDetector
from lib import detection_utils as utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, train_set, epochs, batch_size, test_set=None):
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, collate_fn=lambda x: x)

    for epoch in range(epochs):
        for i_batch, batch in enumerate(train_loader):

            x_batch = torch.tensor([xi['x'] for xi in batch], device=device)
            y_batch = [yi['y'] for yi in batch]

            y_boxes = [utils.labels_to_tensor(yi, H, W) for yi in y_batch]
            detector_out = model(x_batch, y_boxes)

            return


def main():
    train_set = MNISTAugDataset(10)
    test_set = MNISTAugDataset(2, test_mode=True)

    model = MnistDetector()

    epochs = 1
    batch_size = 2

    train(model, train_set, epochs, batch_size, test_set=test_set)


if __name__ == '__main__':
    main()
