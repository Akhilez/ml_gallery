import os
import random

import numpy as np
from skimage.transform import resize

from lib.mnist_aug import caption_rules


class MNISTAug:
    def __init__(self):
        self.dm = DataManager()

        self.scale = 4  # height(out_img) / height(in_image)
        self.overflow = 0.3  # An in_image can't overflow more than 50% out of the image

        self.min_objects = 4
        self.max_objects = 10

        self.scaling_mean = 1.25
        self.scaling_sd = 0.4

        self.spacing = 0.7  # Fraction: distance(c1, c2) / (r1 + r2)

    def get_augmented(self, x: np.ndarray, y: np.ndarray, n_out: int, get_captions: bool = False):
        """

        Parameters
        ----------
        x: a tensor of shape [1000, 28, 28]
        y: a tensor of shape [1000, 1]
        n_out: number of output images

        Returns
        -------
        aug_x: np.ndarray: a tensor of shape [1000, 112, 112]
        aug_y: list of list of dicts: [[{
                    'class': int(np.argmax(y[rand_i])),
                    'class_one_hot': y[rand_i],
                    'x1': rand_x,
                    'y1': rand_y,
                    'x2': rand_x + localized_dim_x,
                    'y2': rand_y + localized_dim_y,
                    'cx': rand_x + localized_dim_x / 2,
                    'cy': rand_y + localized_dim_y / 2,
                    'height': localized_dim_y,
                    'width': localized_dim_x
                }]]

        """

        # x_out = width of output image
        x_out = x.shape[1] * self.scale

        aug_x = np.zeros((n_out, x_out, x_out))
        aug_y = []
        captions = []

        for i in range(n_out):

            n_objects = random.randint(self.min_objects, self.max_objects)
            aug_yi = []
            centers = []
            widths = []

            j = 0
            while j < n_objects:
                rand_i = random.randrange(0, len(x))
                x_in = int(max(0, np.random.normal(self.scaling_mean, self.scaling_sd, 1)) * x.shape[1])
                # x_in = int(random.uniform(self.min_resize, self.max_resize) * x.shape[1])

                try:
                    resized_object = resize(x[rand_i], (x_in, x_in))  # TODO: Find the root cause of this error
                except Exception as e:
                    print(e)
                    j -= 1
                    continue

                attempts = 1
                while attempts < self.max_objects * 10:
                    attempts += 1
                    # rand_x, rand_y are the coordinates of object
                    # rand_x = random * (x_out - (x_in * (1-overflow)))
                    # TODO: This does not take into account the overlap on left and top edge.
                    rand_x = int(random.random() * (x_out - (x_in * (1 - self.overflow))))
                    rand_y = int(random.random() * (x_out - (x_in * (1 - self.overflow))))

                    if len(centers) == 0 or not self.is_overlapping(rand_x, rand_y, x_in, centers, widths):
                        break
                else:
                    continue

                widths.append(x_in)
                centers.append((rand_x + x_in / 2, rand_y + x_in / 2))

                # Clip the H and W of x if it is overflowing.
                localized_dim_x = min(x_out - rand_x, x_in)
                localized_dim_y = min(x_out - rand_y, x_in)
                localized_xi = resized_object[:localized_dim_x, :localized_dim_y]

                aug_x[i][rand_x:rand_x + localized_dim_x, rand_y:rand_y + localized_dim_y] += localized_xi

                aug_yi.append({
                    'id': j,
                    'class': int(np.argmax(y[rand_i])),
                    'class_one_hot': y[rand_i],
                    'x1': rand_x,
                    'y1': rand_y,
                    'x2': rand_x + localized_dim_x,
                    'y2': rand_y + localized_dim_y,
                    'cx': rand_x + localized_dim_x / 2,
                    'cy': rand_y + localized_dim_y / 2,
                    'height': localized_dim_y,
                    'width': localized_dim_x,
                    'type': 'number'
                })

                j += 1

            aug_y.append(aug_yi)
            aug_x[i][aug_x[i] > 1] = 1.0

            # DataManager.plot_num(aug_x[i], aug_yi)
            # DataManager.plot_num(aug_x[i])

            if get_captions:
                captions.append(self.get_captions(aug_yi))

        if get_captions:
            return aug_x, aug_y, captions

        return aug_x, aug_y

    def is_overlapping(self, x, y, width, centers, widths):
        cx, cy = x + width / 2, y + width / 2
        for i in range(len(centers)):
            diameter = (width + widths[i]) * 0.5 * self.spacing
            distance = np.linalg.norm(np.array([cx, cy]) - np.array(centers[i]))

            if distance < diameter:
                return True

        return False

    def get_captions(self, boxes):
        captions = []

        for box in boxes:
            number_caption = self.get_number_caption(box['x1'], box['y1'], box['x2'], box['y2'])
            captions.append({'box_id': box['id'], 'caption': number_caption})

        relationship_boxes = self.get_relationship_boxes(boxes)

        for box in relationship_boxes:
            caption = self.get_relationship_caption(box)
            captions.append({'box_id': box['id'], 'caption': caption})

        return boxes + relationship_boxes, captions

    @staticmethod
    def get_number_caption(x1a, y1a, x2a, y2a):

        areas = []
        for grid_box in caption_rules.grid_boxes:
            x1 = max(x1a, grid_box[0])
            y1 = max(y1a, grid_box[1])
            x2 = max(x2a, grid_box[2])
            y2 = max(y2a, grid_box[3])

            areas.append((x2 - x1) * (y2 - y1))

        argmax = np.argmax(areas)

        grid_box_name = np.random.choice(caption_rules.grid_names[argmax])
        number_name = np.random.choice(caption_rules.class_names[argmax])
        caption = np.random.choice(caption_rules.number_captions)
        caption = caption.format(c=number_name, q=grid_box_name)

        return caption

    def get_relationship_boxes(self, boxes):

        # TODO: Finish up
        return [{
            'id': 0,
            'classes': [2, 3],
            'x1': 0.1,
            'y1': 1.2,
            'x2': 3.2,
            'y2': 2.3,
            'cx': 4.3,
            'cy': 5.4,
            'height': 2.2,
            'width': 2.4,
            'type': 'relationship'
        }]

    def get_relationship_caption(self, box):
        # TODO: Finish up
        return 'dummy string'


class DataManager:
    def __init__(self):
        from mlg.settings import BASE_DIR
        self.dir = f'{BASE_DIR}/data/mnist/numbers'

        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

    def load(self):
        self.load_train()
        self.load_test()

    def load_train(self):
        if os.path.exists(self.dir + '/x_train.npy'):
            self.x_train = np.load(f'{self.dir}/x_train.npy')
            self.y_train = np.load(f'{self.dir}/y_train.npy')
        else:
            self.load_train_from_torch()

    def load_test(self):
        if os.path.exists(self.dir + '/x_test.npy'):
            self.x_test = np.load(f'{self.dir}/x_test.npy')
            self.y_test = np.load(f'{self.dir}/y_test.npy')
        else:
            self.load_test_from_torch()

    def load_train_from_torch(self):
        import torch
        import torchvision
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(self.dir, train=True, download=True,
                                       transform=torchvision.transforms.ToTensor()), shuffle=True)
        x_train = []
        y_train = []

        for data in train_loader:
            x_train.append(data[0].reshape(28, 28).numpy())
            y_train.append(data[1][0])

        self.y_train = torch.tensor(self.to_one_hot(y_train))
        self.x_train = torch.tensor(x_train)

    def load_test_from_torch(self):
        import torch
        import torchvision
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(self.dir, train=False, download=True,
                                       transform=torchvision.transforms.ToTensor()), shuffle=True)
        x_test = []
        y_test = []

        for data in test_loader:
            x_test.append(data[0].reshape(28, 28).numpy())
            y_test.append(data[1][0])

        self.y_test = torch.tensor(self.to_one_hot(y_test))
        self.x_test = torch.tensor(x_test)

    @staticmethod
    def plot_num(x, bounding_boxes: list = None):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.imshow(x, cmap='gray')

        if bounding_boxes is not None:
            import matplotlib.patches as patches

            for i in range(len(bounding_boxes)):
                x1 = bounding_boxes[i]['x1']
                y1 = bounding_boxes[i]['y1']
                x2 = bounding_boxes[i]['x2']
                y2 = bounding_boxes[i]['y2']
                rect = patches.Rectangle((y1, x1), y2 - y1, x2 - x1, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

                if 'class' in bounding_boxes[i]:
                    ax.text(y1, x1, bounding_boxes[i]['class'], size=8, ha="left", va="top",
                            bbox=dict(boxstyle="square", fc=(1., 0.8, 0.8)))

        fig.show()

    @staticmethod
    def one_hot_to_num(x):
        return np.argmax(x)

    @staticmethod
    def to_one_hot(x):
        b = np.zeros((len(x), 10), dtype=np.float32)
        b[np.arange(len(x)), x] = 1
        return b
