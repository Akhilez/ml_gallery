import torch

from app.vision.dense_cap import detection_utils as utils
from lib.mnist_aug.mnist_augmenter import MNISTAug, DataManager

dm = DataManager()
dm.load_test()

aug = MNISTAug()
x, y = aug.get_augmented(dm.x_test, dm.y_test, 2)

x = torch.tensor(x, dtype=torch.float32).view((-1, 1, 112, 112))

H = 14
W = 14
k = 9

i = 0

y_ = utils.labels_to_tensor(y[i], 112, 112)

anchors_tensor = utils.generate_anchors(
    shape=(W, H), sizes=(0.15, 0.45, 0.75), ratios=(0.5, 1, 2)
)

DataManager.plot_num(torch.ones((112, 112)), y[i])

iou = utils.get_iou_map(y_[i], anchors_tensor)

confidences, diffs = utils.get_labels(
    iou, y_[i], anchors_tensor, k, H, W, confidence_threshold=0.7
)

confidences[confidences > 0.7]
