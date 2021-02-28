import albumentations as A
import cv2
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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import optuna

from settings import BASE_DIR

height = 160
width = 224

writer: SummaryWriter = None


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
        # A.RandomSizedCrop(min_max_height=(100, 150), height=height, width=width, p=0.5),
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
        mask = np.array([mask])

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
test_images_path = f"{data_dir}/testing_set/images"
test_masks_path = f"{data_dir}/testing_set/masks"

dataset = IrisImageDataset(
    images_path=train_images_path, masks_path=training_masks_path, transform=transform
)
len_dataset = len(dataset)
test_dataset = IrisImageDataset(
    images_path=test_images_path, masks_path=test_masks_path, transform=transform
)


class IrisGenerator(nn.Module):
    def __init__(self):
        super(IrisGenerator, self).__init__()

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


class IrisCritic(nn.Module):
    def __init__(self):
        super(IrisCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Conv2d(6, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=5, stride=5, padding=1),
        )

    def forward(self, x):
        return torch.sigmoid(self.critic(x)).flatten(1).mean(1)


model: IrisGenerator = None
critic: IrisCritic = None

optim: torch.optim.Adam = None
optim_critic: torch.optim.Adam = None

criterion_cross_entropy = nn.CrossEntropyLoss(reduction="none")

config: DictConfig = None


def plot_one_hot_mask(mask):
    mask = mask.argmax(0) / 2
    plt.imshow(mask, cmap="gray")
    plt.show()


def get_one_hot_masks(masks):
    batch_size, _, h, w = masks.shape
    one_hot_masks = torch.zeros((batch_size, 3, h, w))
    one_hot_masks[:, 0, :, :][masks[:, 0, :, :] == 0] = 1
    one_hot_masks[:, 1, :, :][masks[:, 0, :, :] == 1] = 1
    one_hot_masks[:, 2, :, :][masks[:, 0, :, :] == 2] = 1
    return one_hot_masks


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


def train_critic(images, masks):
    optim_critic.zero_grad()

    # ---- Real labels ----
    real_labels = torch.ones((len(images),))

    critic_out = critic(torch.cat((images, get_one_hot_masks(masks)), 1))
    loss_real = F.binary_cross_entropy(critic_out, real_labels)

    # ---- Train with fakes ---
    generated = model(images)
    critic_out = critic(torch.cat((images, generated), 1))
    loss_fake = F.binary_cross_entropy(critic_out, real_labels * 0)

    # --- optimize ----

    loss = config.lr_critic_real * loss_real + config.lr_critic_fake * loss_fake
    loss.backward()
    optim_critic.step()

    return torch.stack((loss_real, loss_fake)).tolist()


def train_gen(images, masks):
    optim.zero_grad()
    generated = model(images)
    critic_out = critic(torch.cat((images, generated), 1))

    loss_entropy = criterion_cross_entropy(
        generated, masks.type(torch.LongTensor).squeeze(1)
    )
    loss_entropy = torch.mean(loss_entropy * get_weight_map(masks))
    loss_gen = F.binary_cross_entropy(critic_out, torch.ones((len(images),)))

    loss = config.lr_entropy * loss_entropy + config.lr_gen * loss_gen
    loss.backward()
    optim.step()

    return torch.stack((loss_entropy, loss_gen)).tolist()


def train(cfg, trail):
    global optim, optim_critic, config, writer, model, critic

    # cfg.lr_critic_real = trail.suggest_loguniform("lr_critic_real", 0.0001, 1)
    # cfg.lr_critic_fake = trail.suggest_loguniform("lr_critic_fake", 0.0001, 1)
    # cfg.lr_entropy = trail.suggest_loguniform("lr_entropy", 0.0001, 1)
    # cfg.lr_gen = trail.suggest_loguniform("lr_gen", 0.0001, 1)

    config = cfg

    writer = SummaryWriter(
        f"{BASE_DIR}/app/vision/iris_detector/L2/runs/iris_real_{cfg.lr_critic_real}_fake_{cfg.lr_critic_fake}_entropy_{cfg.lr_entropy}_gen_{cfg.lr_gen}_{int(datetime.now().timestamp())}"
    )

    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    global_step = 0

    model = IrisGenerator()
    critic = IrisCritic()

    optim = torch.optim.Adam(model.parameters())
    optim_critic = torch.optim.Adam(critic.parameters())

    loss_entropy, loss_gen, loss_real, loss_fake = (0, 0, 0, 0)

    for epoch in range(config.epochs):

        for batch_i, (images, masks) in enumerate(train_loader):
            global_step += 1

            loss_entropy, loss_gen = train_gen(images, masks)
            loss_real, loss_fake = train_critic(images, masks)

            writer.add_scalar("loss_real", loss_real, global_step=global_step)
            writer.add_scalar("loss_fake", loss_fake, global_step=global_step)
            writer.add_scalar("loss_entropy", loss_entropy, global_step=global_step)
            writer.add_scalar("loss_gen", loss_gen, global_step=global_step)

            print(".", end="")
        print()

        _, (test_images, _) = next(enumerate(test_loader))
        with torch.no_grad():
            test_generated = model(test_images)
        plot_one_hot_mask(test_generated[0])

        # y = model(images)
        #
        # loss = criterion(y, masks.type(torch.LongTensor))
        # loss = torch.mean(loss * get_weight_map(masks))
        #
        # optim.zero_grad()
        # loss.backward()
        # optim.step()

        # plot_one_hot_mask(y)

    writer.close()
    final_loss = float(np.mean([loss_entropy, loss_gen, loss_real, loss_fake]))
    writer.add_hparams(
        {
            key: config[key]
            for key in ["lr_critic_real", "lr_critic_fake", "lr_entropy", "lr_gen"]
        },
        {"final_loss": final_loss},
    )
    return final_loss


@hydra.main(config_name="config")
def main(cfg: DictConfig) -> None:
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trail: train(cfg, trail), n_trials=1)
    print(f"{study.best_params=}")
    print(f"{study.best_value=}")


if __name__ == "__main__":
    # main(DictConfig({"epochs": 1, "batch_size": 10}))
    main()
