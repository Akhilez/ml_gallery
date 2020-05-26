import numpy as np
import random
from skimage.transform import resize


class MNISTAug:
    def __init__(self):
        self.dm = DataManager()

        self.scale = 4  # height(out_img) / height(in_image)
        self.overflow = 0.3  # An in_image can't overflow more than 50% out of the image

        self.min_objects = 4
        self.max_objects = 10

        self.min_resize = 0.2  # The smallest object size in the out_image WRT original object size
        self.max_resize = 3

        self.spacing = 1  # Fraction: distance(c1, c2) / (r1 + r2)

    def get_augmented(self, x: np.ndarray, y: np.ndarray, n_out: int):
        """

        Parameters
        ----------
        x: a tensor of shape [1000, 28, 28]
        y: a tensor of shape [1000, 1]
        n_out: number of output images

        Returns
        -------
        aug_x: np.ndarray: a tensor of shape [1000, 112, 112]
        aug_y: list: a tensor of shape [n_out, numbers_out, 5] | 5 => [class, x1, y1, x2, y2]

        """

        # x_out = width of output image
        x_out = x.shape[1] * self.scale

        aug_x = np.zeros((n_out, x_out, x_out))
        aug_y = []

        for i in range(n_out):

            n_objects = random.randint(self.min_objects, self.max_objects)
            aug_yi = []

            for j in range(n_objects):

                rand_i = random.randrange(0, len(x))
                x_in = int(max(0, np.random.normal(1, 0.25, 1)) * x.shape[1])
                # x_in = int(random.uniform(self.min_resize, self.max_resize) * x.shape[1])

                resized_object = resize(x[rand_i], (x_in, x_in))

                # rand_x, rand_y are the coordinates of object
                # rand_x = random * (x_out - (x_in * (1-overflow)))
                # TODO: This does not take into account the overlap on left and top edge.
                rand_x = int(random.random() * (x_out - (x_in * (1-self.overflow))))
                rand_y = int(random.random() * (x_out - (x_in * (1-self.overflow))))

                # TODO: Add spacing to rand_x and rand_y

                # Clip the H and W of x if it is overflowing.
                localized_dim_x = min(x_out - rand_x, x_in)
                localized_dim_y = min(x_out - rand_y, x_in)
                localized_xi = resized_object[:localized_dim_x, :localized_dim_y]

                aug_x[i][rand_x:rand_x + localized_dim_x, rand_y:rand_y + localized_dim_y] += localized_xi

                aug_yi.append([
                    y[rand_i],  # Class of i
                    rand_x,
                    rand_y,
                    rand_x + localized_dim_x,
                    rand_y + localized_dim_y
                ])

            aug_y.append(aug_yi)

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
        import matplotlib.pyplot as plt
        plt.imshow(x, cmap='gray')
        plt.show()
