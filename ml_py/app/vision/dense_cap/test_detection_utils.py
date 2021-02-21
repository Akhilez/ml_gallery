import unittest
import torch

from app.vision.dense_cap.detection_test import get_iou_map


class TestDetectionUtils(unittest.TestCase):
    def test_get_iou_map_1(self):
        b1 = torch.FloatTensor([[3, 3, 4, 4], [7, 7, 4, 4]]).T / 10

        b2 = torch.FloatTensor([[3, 5, 2, 2], [7, 7, 2, 2]]).T / 10

        iou = get_iou_map(b1, b2)

        iou_label = torch.FloatTensor([[1 / 9, 0], [0, 0.25]])

        diff = float(torch.abs(iou - iou_label).max())

        self.assertGreater(0.00001, diff)
