from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.optim import Adam, RMSprop
from torchvision import ops
from torch.nn import functional as F

from lib.mnist_aug.loader import MNISTAugDataset
from app.vision.find_all_chars.model import MnistDetector
from lib import detection_utils as utils
from lib.mnist_aug.mnist_augmenter import DataManager, MNISTAug

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_labels(model, detector_out, x_batch, y_boxes):
    # Shape: (batch, k, H, W) | ones and zeros tensor.
    confidences_labels = utils.get_confidences(
        torch.stack(detector_out.iou_max)
        if len(detector_out.iou_max) > 0
        else torch.empty([]),
        model.threshold_p,
        (len(x_batch), model.k, model.Hp, model.Wp),
    )

    diffs_labels = torch.stack(
        [
            utils.get_diffs(
                y_boxes[j_batch],
                model.anchors_tensor,
                detector_out.iou_max[j_batch],
                detector_out.matched_bboxes[j_batch],
                model.k,
                model.Hp,
                model.Wp,
            )  # Shape: (4, k, H, W)
            for j_batch in range(len(x_batch))
        ]
    )

    return confidences_labels, diffs_labels


def get_loss(detector_out, confidences_labels, diffs_labels, get_all=False):
    confidences_loss = F.binary_cross_entropy(
        detector_out.confidences, confidences_labels
    )

    diffs_loss = []

    for i_batch in range(len(detector_out.diffs)):
        idx = (
            torch.cat((detector_out.idx_p[i_batch], detector_out.idx_n[i_batch]))
            if len(detector_out.idx_n) > 0
            else detector_out.idx_p[i_batch]
        )
        not_nan_idx = diffs_labels[i_batch].flatten(0)[idx].isnan() == False

        diffs_pred = detector_out.diffs[i_batch].flatten(0)[idx][not_nan_idx]
        diffs_labels_ = diffs_labels[i_batch].flatten(0)[idx][not_nan_idx]

        diffs_loss.append(F.mse_loss(diffs_pred, diffs_labels_))

    diffs_loss = sum(diffs_loss)

    if get_all:
        return confidences_loss, diffs_loss

    total_loss = confidences_loss + diffs_loss
    return total_loss


def train(model, train_set, epochs, batch_size, test_set=None):
    train_loader = DataLoader(
        train_set, shuffle=True, batch_size=batch_size, collate_fn=lambda x: x
    )

    model.train()
    optimizer = Adam(model.parameters())
    # optimizer = RMSprop(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch}\n")
        for i_batch, batch in enumerate(train_loader):

            x_batch = torch.stack([xi["x"] for xi in batch])
            y_batch = [yi["y"] for yi in batch]

            y_boxes = [utils.labels_to_tensor(yi, model.H, model.W) for yi in y_batch]
            detector_out = model(x_batch, y_boxes)

            confidences_labels, diffs_labels = get_labels(
                model, detector_out, x_batch, y_boxes
            )

            confidence_loss, diff_loss = get_loss(
                detector_out, confidences_labels, diffs_labels, get_all=True
            )
            loss = confidence_loss + diff_loss

            print(confidence_loss.item(), diff_loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test(model, dataset):
    # type: (nn.Module, MNISTAugDataset) -> None

    batch_size = 10

    test_loader = DataLoader(
        dataset, shuffle=True, batch_size=batch_size, collate_fn=lambda x: x
    )

    model.eval()

    with torch.no_grad():

        for i_batch, batch in enumerate(test_loader):
            x_batch = torch.stack([xi["x"] for xi in batch])
            y_batch = [yi["y"] for yi in batch]

            y_boxes = [utils.labels_to_tensor(yi, model.H, model.W) for yi in y_batch]
            detector_out = model(x_batch, y_boxes)

            confidences_labels, diffs_labels = get_labels(
                model, detector_out, x_batch, y_boxes
            )

            loss = get_loss(detector_out, confidences_labels, diffs_labels)

            print(loss.item())

            nms_boxes = []
            for j_batch in range(len(x_batch)):
                # pred_boxes = torch.cat((detector_out.pred_bbox_n[j_batch].T, detector_out.pred_bbox_p[j_batch].T)).T
                pred_boxes = detector_out.pred_bbox_p[j_batch]

                confidences_batch = detector_out.confidences[j_batch].flatten(0)
                confidences_batch = confidences_batch[detector_out.idx_p[j_batch]]
                # confidences_batch_n = confidences_batch[detector_out.idx_n[j_batch]]
                # confidences_batch = torch.cat((confidences_batch_n, confidences_batch_p))

                # De-Normalize to the feature map size
                multiplier = torch.tensor([model.W, model.H, model.W, model.H]).view(
                    (4, 1)
                )
                pred_boxes = (
                    pred_boxes * multiplier
                ).round()  # .type(torch.int32)  # shape (4, p) (x1y1x2y2)

                # remove small boxes
                pred_boxes = model.remove_tiny_boxes(pred_boxes, min_side=5)

                nms_indices = ops.nms(pred_boxes.T, confidences_batch, 0.1)
                nms_boxes_i = pred_boxes[:, nms_indices]

                print(nms_boxes_i.shape)
                nms_boxes.append(utils.tensor_to_labels(nms_boxes_i))

                DataManager.plot_num(
                    x_batch[j_batch].view((model.H, model.W)), nms_boxes[j_batch]
                )


def main():

    aug = MNISTAug()
    aug.min_objects = 5
    aug.max_objects = 9

    train_set = MNISTAugDataset(1000, aug=aug)
    test_set = MNISTAugDataset(2, test_mode=True, aug=aug)

    model = MnistDetector()

    epochs = 2
    batch_size = 32

    train(model, train_set, epochs, batch_size, test_set=test_set)
    test(model, test_set)


if __name__ == "__main__":
    main()
