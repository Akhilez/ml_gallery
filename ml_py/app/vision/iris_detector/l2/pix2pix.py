import json

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from matplotlib.patches import Ellipse
import os
import hydra
from omegaconf import DictConfig
import copy

from settings import BASE_DIR

height = 160
width = 224


def plot_img(image, ellipses=None, show=False):
    """
    image: np.array of shape (c, h, w)
    ellipses: np.array of shape (n, 5)
    """
    plt.imshow(np.moveaxis(np.array(image), 0, -1))

    _, h, w = image.shape

    if ellipses is not None and len(ellipses) > 0:
        for ellipse in ellipses:
            xc, yc, rx, ry, a = ellipse
            plt.gca().add_patch(
                Ellipse(xy=(xc, yc), width=2 * rx, height=2 * ry, angle=a, fill=False)
            )

    if show:
        plt.show()


transform = A.Compose(
    [
        A.Resize(height=height, width=width),
        # A.RandomSizedCrop(min_max_height=(250, 250), height=300, width=400, p=0.5),
        # A.CenterCrop(height=200, width=200),
        # A.ToGray(p=0.2),
        # A.ChannelDropout(channel_drop_range=(1, 2), p=0.2),
        # A.ChannelShuffle(p=0.2),
        # A.HueSaturationValue(p=0.2),
        # A.ImageCompression(quality_lower=60, p=0.1),
        # A.Posterize(p=0.2),
        # A.Rotate(limit=40, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        # A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
    ],
    # keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
)


class IrisImageDataset(Dataset):
    def __init__(self, images_path, masks_path, labels_path=None, transform=None):
        super(IrisImageDataset, self).__init__()
        self.data = []
        self.images_path = images_path
        self.labels_path = labels_path
        self.masks_path = masks_path
        self.transform = transform

        self.image_names = self.get_images_list(masks_path, file_ext=".png")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(f"{self.images_path}/{image_name}.png")
        image = np.array(image)

        mask = Image.open(f"{self.masks_path}/{image_name}.png")
        mask = np.array(mask)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # Covert from channels last to channels first
        image = np.moveaxis(image, -1, 0)
        # mask = np.moveaxis(mask, -1, 0)

        return image, mask

    @staticmethod
    def get_images_list(images_dir, file_ext=None):
        files_list = sorted(os.listdir(images_dir))
        extension_len = len(file_ext)
        if file_ext:
            file_list_ = []
            for file_name in files_list:
                if file_name[-extension_len:] == file_ext:
                    file_list_.append(file_name[:-extension_len])
            files_list = file_list_
        return files_list


data_dir = f"{BASE_DIR}/data/pupil/L2"
train_images_path = f"{data_dir}/training_set/images"
training_labels_path = f"{data_dir}/training_set/ground_truth"
training_masks_path = f"{data_dir}/training_set/masks"

dataset = IrisImageDataset(
    images_path=train_images_path, masks_path=training_masks_path, transform=transform
)


class IrisUNet(nn.Module):
    def __init__(self):
        super(IrisUNet, self).__init__()

        self.modules = [
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1),  # 6
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # 11
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
        ]
        self.modules_list = nn.ModuleList(self.modules)

    def forward(self, x):
        input_shape = x.shape
        compressions = []

        for i in range(6):
            x = self.modules[i](x)
            x = torch.relu(x)
            compressions.append(x)

        x = F.relu(
            self.modules[6](compressions.pop(), output_size=compressions[-2].shape)
        )
        x = F.relu(self.modules[7](x + compressions.pop()))

        x = F.relu(
            self.modules[8](x + compressions.pop(), output_size=compressions[-2].shape)
        )
        x = F.relu(self.modules[9](x + compressions.pop()))

        x = F.relu(self.modules[10](x + compressions.pop(), output_size=input_shape))
        x = F.relu(self.modules[11](x + compressions.pop()))

        x = F.relu(self.modules[12](x))
        x = self.modules[13](x)

        return x


model = IrisUNet()

optim = torch.optim.Adam(model.parameters())


def plot_one_hot_mask(mask):
    mask = mask.argmax(1) / 2
    print(mask.max())
    print(mask.min())
    plt.imshow(mask[0], cmap="gray")
    plt.show()


def get_class_weights():

    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    n_bg = []
    n_iris = []
    n_pupil = []

    for images, masks in loader:
        n_bg.append(len(torch.nonzero((masks == 0).flatten())) / len(masks))
        n_iris.append(len(torch.nonzero((masks == 2).flatten())) / len(masks))
        n_pupil.append(len(torch.nonzero((masks == 1).flatten())) / len(masks))

    n_bg = np.mean(n_bg)
    n_iris = np.mean(n_iris)
    n_pupil = np.mean(n_pupil)

    cls = torch.tensor([n_bg, n_iris, n_pupil])

    weights = 1 - (cls / torch.sum(cls))

    return weights  # [0.1770, 0.8455, 0.9775]


def get_weight_map(masks):
    weights = [0.1770, 0.9775, 0.8455]
    weight_map = copy.deepcopy(masks).type(torch.FloatTensor)
    weight_map[weight_map == 0] = weights[0]
    weight_map[weight_map == 1] = weights[1]
    weight_map[weight_map == 2] = weights[2]
    return weight_map


def train(config):

    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss(reduction="none")

    for epoch in range(config.epochs):
        y = None

        for images, masks in train_loader:

            y = model(images)
            loss = criterion(y, masks.type(torch.LongTensor))
            loss = torch.mean(loss * get_weight_map(masks))

            optim.zero_grad()
            loss.backward()
            optim.step()

            print(loss.item())

        plot_one_hot_mask(y)


@hydra.main(config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
