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

        self.closeness_fraction = 0  # more = only close ones have relationship

    def get_augmented(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_out: int,
        noisy: bool = False,
        get_class_captions: bool = False,
        get_relationships: bool = False,
        get_positional_labels: bool = False,
        get_positional_relationships: bool = False,
        get_relationship_captions: bool = False,
    ):
        """

        Parameters
        ----------
        x: a tensor of shape [1000, 28, 28]
        y: a tensor of shape [1000, 1]
        n_out: number of output images
        noisy: bool: will add patchy perlin noise to the image # TODO: Lets add some noise to the image
        get_class_captions: bool: will return captions for each number in the image
        get_relationships: bool: will return labels with relationship b/w close numbers
        get_positional_labels: bool: will return labels with positional data. For ex: "2 is in the top right"
        get_positional_relationships: bool: will return relationships with positional data. Ex: "2 is left of 3"
        get_relationship_captions: bool: will return captions for each image.

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

        for i in range(n_out):

            n_objects = random.randint(self.min_objects, self.max_objects)
            aug_yi = []
            centers = []
            widths = []

            j = 0
            while j < n_objects:
                rand_i = random.randrange(0, len(x))
                x_in = int(
                    max(0, np.random.normal(self.scaling_mean, self.scaling_sd, 1))
                    * x.shape[1]
                )
                # x_in = int(random.uniform(self.min_resize, self.max_resize) * x.shape[1])

                try:
                    resized_object = resize(
                        x[rand_i], (x_in, x_in)
                    )  # TODO: Find the root cause of this error
                except Exception as e:
                    print(e)
                    continue

                attempts = 1
                while attempts < self.max_objects * 10:
                    attempts += 1
                    # rand_x, rand_y are the coordinates of object
                    # rand_x = random * (x_out - (x_in * (1-overflow)))
                    # TODO: This does not take into account the overlap on left and top edge.
                    rand_x = int(
                        random.random() * (x_out - (x_in * (1 - self.overflow)))
                    )
                    rand_y = int(
                        random.random() * (x_out - (x_in * (1 - self.overflow)))
                    )

                    if len(centers) == 0 or not self.is_overlapping(
                        rand_x, rand_y, x_in, centers, widths
                    ):
                        break
                else:
                    j += 1
                    continue

                widths.append(x_in)
                centers.append((rand_x + x_in / 2, rand_y + x_in / 2))

                # Clip the H and W of x if it is overflowing.
                localized_dim_x = min(x_out - rand_x, x_in)
                localized_dim_y = min(x_out - rand_y, x_in)
                localized_xi = resized_object[:localized_dim_x, :localized_dim_y]

                aug_x[i][
                    rand_x : rand_x + localized_dim_x, rand_y : rand_y + localized_dim_y
                ] += localized_xi

                aug_yi.append(
                    {
                        "id": j,
                        "class": int(np.argmax(y[rand_i])),
                        "class_one_hot": y[rand_i],
                        "x1": rand_x,
                        "y1": rand_y,
                        "x2": rand_x + localized_dim_x,
                        "y2": rand_y + localized_dim_y,
                        "cx": rand_x + localized_dim_x / 2,
                        "cy": rand_y + localized_dim_y / 2,
                        "height": localized_dim_y,
                        "width": localized_dim_x,
                        "type": "number",
                    }
                )

                j += 1

            aug_x[i][aug_x[i] > 1] = 1.0

            if get_positional_labels:
                aug_yi = self.get_positional_labels(aug_yi)
                if get_class_captions:
                    aug_yi = self.get_positional_captions(aug_yi)
            elif get_class_captions:
                aug_yi = self.get_class_captions(aug_yi)

            if get_relationships:
                rel_boxes = self.get_relationship_boxes(aug_yi)
                if get_positional_relationships:
                    rel_boxes = self.get_positional_relationships(rel_boxes)
                    if get_relationship_captions:
                        rel_boxes = self.get_positional_relationship_captions(rel_boxes)
                elif get_relationship_captions:
                    rel_boxes = self.get_relationship_captions(rel_boxes)
                aug_yi += rel_boxes

            aug_y.append(aug_yi)

        return aug_x, aug_y

    def is_overlapping(self, x, y, width, centers, widths):
        cx, cy = x + width / 2, y + width / 2
        for i in range(len(centers)):
            diameter = (width + widths[i]) * 0.5 * self.spacing
            distance = np.linalg.norm(np.array([cx, cy]) - np.array(centers[i]))

            if distance < diameter:
                return True

        return False

    @staticmethod
    def get_positional_labels(boxes):
        """
        For each number in the image:
            Find the position
            Add position KV to boxes
        return boxes
        """

        for box in boxes:
            box["position"] = MNISTAug.get_number_position(
                box["x1"], box["y1"], box["x2"], box["y2"]
            )
            box["position_one_hot"] = DataManager.to_one_hot([box["position"]], 9)

        return boxes

    @staticmethod
    def get_positional_captions(boxes):
        """
        For each box in boxes:
            generate a random positional caption for position KV
            Add class to number key
            Add positional caption into class key
        return boxes
        """
        for box in boxes:
            grid_box_name = np.random.choice(caption_rules.grid_names[box["position"]])
            number_name = np.random.choice(caption_rules.class_names[box["class"]])
            caption = np.random.choice(caption_rules.positional_captions)
            box["caption"] = caption.format(a=number_name, p=grid_box_name)

        return boxes

    @staticmethod
    def get_class_captions(boxes):
        """
        For each box:
            generate a random number caption
            move class KV to number
            add caption as class key
        """

        for box in boxes:
            box["class"] = np.random.choice(caption_rules.number_captions).format(
                a=box["class"]
            )

        return boxes

    def get_relationship_boxes(self, boxes):
        """
        Find close together pairs
        For each pair:
            Add the numbers into 'close_paris' key
            Add the relationship bounding box coordinates
            Add both number's bounding box coordinates
        """

        relationship_boxes = []
        done_pairs = []
        box_id = len(boxes)

        for i in range(len(boxes)):
            done = False
            distance_factors = []
            for j in range(len(boxes)):
                if i == j:
                    distance_factors.append(9999)
                    continue

                if MNISTAug.is_pair_done(i, j, done_pairs):
                    done = True
                    break

                c1 = np.array(
                    (
                        boxes[i]["cx"],
                        boxes[i]["cy"],
                    )
                )
                c2 = np.array(
                    (
                        boxes[j]["cx"],
                        boxes[j]["cy"],
                    )
                )

                distance = np.linalg.norm(c2 - c1)
                side = boxes[i]["width"] + boxes[j]["width"]

                distance_factor = (side - distance) / side
                if distance_factor < self.closeness_fraction:
                    distance_factor = 9999

                distance_factors.append(distance_factor)

            if done:
                continue

            arg_min = int(np.argmin(distance_factors))
            if distance_factors[arg_min] == 9999:
                continue

            done_pairs.append(
                {"box1": i, "box2": arg_min, "dist": distance_factors[arg_min]}
            )

            relationship_box = MNISTAug.get_relationship_bounding_box(
                boxes[i], boxes[arg_min]
            )
            relationship_box["id"] = box_id
            box_id += 1

            relationship_boxes.append(relationship_box)

        return relationship_boxes

    @staticmethod
    def get_positional_relationships(boxes):
        """
        TODO: Complete positional relationships
        For each box:
            Get the relationship pairs
            Find the positional relationship
            add it to the boxes
        """
        return boxes

    @staticmethod
    def get_positional_relationship_captions(boxes):
        """
        TODO: Incomplete
        For each positional relationship:
            Get a random caption
            Add it to the boxes
        """
        return boxes

    @staticmethod
    def get_relationship_captions(boxes):
        """
        For each relationship:
            get a random caption
            add it to the boxes
        """
        for box in boxes:
            number1_name = np.random.choice(
                caption_rules.class_names[box["box1"]["class"]]
            )
            number2_name = np.random.choice(
                caption_rules.class_names[box["box2"]["class"]]
            )

            caption = np.random.choice(caption_rules.relationship_captions).format(
                a=number1_name, b=number2_name
            )
            box["caption"] = caption

        return boxes

    @staticmethod
    def get_number_position(x1a, y1a, x2a, y2a):

        areas = []
        for grid_box in caption_rules.grid_boxes:
            if (
                x2a < grid_box[0]
                or x1a > grid_box[2]
                or y1a > grid_box[3]
                or y2a < grid_box[1]
            ):
                areas.append(0)
                continue

            x1 = max(x1a, grid_box[0])
            y1 = max(y1a, grid_box[1])
            x2 = min(x2a, grid_box[2])
            y2 = min(y2a, grid_box[3])

            areas.append((x2 - x1) * (y2 - y1))

        argmax = np.argmax(areas)

        return argmax

    @staticmethod
    def get_relationship_bounding_box(box1, box2):
        x1 = min(box1["x1"], box2["x1"])
        y1 = min(box1["y1"], box2["y1"])
        x2 = max(box1["x2"], box2["x2"])
        y2 = max(box1["y2"], box2["y2"])

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        w = x2 - x1
        h = y2 - y1

        return {
            "classes": [box1["class"], box2["class"]],
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": cx,
            "cy": cy,
            "height": h,
            "width": w,
            "type": "relationship",
            "box1": box1,
            "box2": box2,
        }

    @staticmethod
    def is_pair_done(i, j, done_pairs):
        for pair in done_pairs:
            if pair["box1"] == i and pair["box2"] == j:
                return True
            if pair["box1"] == j and pair["box2"] == i:
                return True
        return False


class DataManager:
    def __init__(self):
        from settings import BASE_DIR

        self.dir = f"{BASE_DIR}/data/mnist/numbers"

        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

    def load(self):
        self.load_train()
        self.load_test()

    def load_train(self):
        if os.path.exists(self.dir + "/x_train.npy"):
            self.x_train = np.load(f"{self.dir}/x_train.npy")
            self.y_train = np.load(f"{self.dir}/y_train.npy")
        else:
            self.load_train_from_torch()

    def load_test(self):
        if os.path.exists(self.dir + "/x_test.npy"):
            self.x_test = np.load(f"{self.dir}/x_test.npy")
            self.y_test = np.load(f"{self.dir}/y_test.npy")
        else:
            self.load_test_from_torch()

    def load_train_from_torch(self):
        import torch
        import torchvision

        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                self.dir,
                train=True,
                download=True,
                transform=torchvision.transforms.ToTensor(),
            ),
            shuffle=True,
        )
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
            torchvision.datasets.MNIST(
                self.dir,
                train=False,
                download=True,
                transform=torchvision.transforms.ToTensor(),
            ),
            shuffle=True,
        )
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

        ax.imshow(x, cmap="gray")

        if bounding_boxes is not None:
            import matplotlib.patches as patches

            for i in range(len(bounding_boxes)):
                x1 = bounding_boxes[i]["x1"]
                y1 = bounding_boxes[i]["y1"]
                x2 = bounding_boxes[i]["x2"]
                y2 = bounding_boxes[i]["y2"]
                rect = patches.Rectangle(
                    (y1, x1),
                    y2 - y1,
                    x2 - x1,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)

                if "class" in bounding_boxes[i]:
                    ax.text(
                        y1,
                        x1,
                        bounding_boxes[i]["class"],
                        size=8,
                        ha="left",
                        va="top",
                        bbox=dict(boxstyle="square", fc=(1.0, 0.8, 0.8)),
                    )

        fig.show()

    @staticmethod
    def one_hot_to_num(x):
        return np.argmax(x)

    @staticmethod
    def to_one_hot(x, num_classes=10):
        b = np.zeros((len(x), num_classes), dtype=np.float32)
        b[np.arange(len(x)), x] = 1
        return b
