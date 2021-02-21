"""
1. Rescale mnist data to VGG input size
2. Get VGG model weights.
3. Add 2D SOFM layers after 7x7 conv
4. Add 1D SOFM layer for final clustering.
"""

import numpy as np
from skimage.transform import resize

data_path = "data/cifar10"
data = np.load(f"{data_path}/data_batch_1")

resize(images[rand_i], (224, 224))
