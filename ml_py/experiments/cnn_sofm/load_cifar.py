import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os

"""

Copied from: https://github.com/snatch59/load-cifar-10/blob/master/load_cifar_10_alt.py

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 
training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains 
exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random 
order, but some training batches may contain more images from one class than another. Between them, the training 
batches contain exactly 5000 images from each class.
"""


def load_batch(f_path, label_key="labels"):
    """Internal utility for parsing CIFAR data.

    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    # Returns
        A tuple `(data, labels)`.
    """
    with open(f_path, "rb") as f:
        if sys.version_info < (3,):
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding="bytes")
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode("utf8")] = v
            d = d_decoded
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_data(path, negatives=False):
    """Loads CIFAR10 dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    num_train_samples = 50000

    x_train_local = np.empty((num_train_samples, 3, 32, 32), dtype="uint8")
    y_train_local = np.empty((num_train_samples,), dtype="uint8")

    for i in range(1, 6):
        fpath = os.path.join(path, "data_batch_" + str(i))
        (
            x_train_local[(i - 1) * 10000 : i * 10000, :, :, :],
            y_train_local[(i - 1) * 10000 : i * 10000],
        ) = load_batch(fpath)

    fpath = os.path.join(path, "test_batch")
    x_test_local, y_test_local = load_batch(fpath)

    y_train_local = np.reshape(y_train_local, (len(y_train_local), 1))
    y_test_local = np.reshape(y_test_local, (len(y_test_local), 1))

    if negatives:
        x_train_local = x_train_local.transpose(0, 2, 3, 1).astype(np.float32)
        x_test_local = x_test_local.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        x_train_local = np.rollaxis(x_train_local, 1, 4)
        x_test_local = np.rollaxis(x_test_local, 1, 4)

    return (x_train_local, y_train_local), (x_test_local, y_test_local)


if __name__ == "__main__":
    """show it works"""

    cifar_10_dir = "../../data/cifar10"

    (x_train, y_train), (x_test, y_test) = load_data(cifar_10_dir)

    print("Train data (x_train): ", x_train.shape)
    print("Train labels (y_train): ", y_train.shape)
    print("Test data (x_test): ", x_test.shape)
    print("Test labels (y_test): ", y_test.shape)

    # Don't forget that the label_names and filesnames are in binary and need conversion if used.

    # display some random training images in a 25x25 grid
    num_plot = 5
    fig, ax = plt.subplots(num_plot, num_plot)
    for m in range(num_plot):
        for n in range(num_plot):
            idx = np.random.randint(0, x_train.shape[0])
            ax[m, n].imshow(x_train[idx])
            ax[m, n].get_xaxis().set_visible(False)
            ax[m, n].get_yaxis().set_visible(False)
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0)
    plt.show()
