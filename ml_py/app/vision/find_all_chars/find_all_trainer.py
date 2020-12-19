from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.optim import Adam

from lib.mnist_aug.loader import MNISTAugDataset
from app.vision.find_all_chars.model import MnistDetector
from lib import detection_utils as utils
from lib.mnist_aug.mnist_augmenter import DataManager

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, train_set, epochs, batch_size, test_set=None):
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, collate_fn=lambda x: x)

    confidences_loss_fn = nn.BCELoss()
    diffs_loss_fn = nn.L1Loss()

    model.train()
    optimizer = Adam(model.parameters())

    for epoch in range(epochs):
        for i_batch, batch in enumerate(train_loader):

            x_batch = torch.tensor([xi['x'] for xi in batch], device=device)
            y_batch = [yi['y'] for yi in batch]

            y_boxes = [utils.labels_to_tensor(yi, model.H, model.W) for yi in y_batch]
            detector_out = model(x_batch, y_boxes)

            # Shape: (batch, k, H, W) | ones and zeros tensor.
            confidences_labels = utils.get_confidences(
                torch.stack(detector_out.iou_max),
                model.threshold_p,
                (batch_size, model.k, model.Hp, model.Wp)
            )

            diffs_labels = torch.stack([
                utils.get_diffs(
                    y_boxes[i_batch],
                    model.anchors_tensor,
                    detector_out.iou_max[i_batch],
                    detector_out.matched_bboxes[i_batch],
                    model.k,
                    model.Hp,
                    model.Wp
                )  # Shape: (4, k, H, W)
                for i_batch in range(batch_size)
            ])

            confidences_loss = confidences_loss_fn(detector_out.confidences, confidences_labels)
            diffs_loss = diffs_loss_fn(detector_out.diffs, diffs_labels)

            total_loss = confidences_loss + diffs_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


def test(model, dataset):
    # type: (nn.Module, MNISTAugDataset) -> None

    batch_size = 10

    test_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=lambda x: x)

    model.eval()

    with torch.no_grad():

        for i_batch, batch in enumerate(test_loader):
            x_batch = torch.tensor([xi['x'] for xi in batch], device=device)
            y_batch = [yi['y'] for yi in batch]

            y_boxes = [utils.labels_to_tensor(yi, model.H, model.W) for yi in y_batch]
            detector_out = model(x_batch, y_boxes)

            # Shape: (batch, k, H, W) | ones and zeros tensor.
            confidences_labels = utils.get_confidences(
                torch.stack(detector_out.iou_max),
                model.threshold_p,
                (batch_size, model.k, model.Hp, model.Wp)
            )

            diffs_labels = torch.stack([
                utils.get_diffs(
                    y_boxes[i_batch],
                    model.anchors_tensor,
                    detector_out.iou_max[i_batch],
                    detector_out.matched_bboxes[i_batch],
                    model.k,
                    model.Hp,
                    model.Wp
                )  # Shape: (4, k, H, W)
                for i_batch in range(batch_size)
            ])

            confidences_loss_fn = nn.BCELoss()
            diffs_loss_fn = nn.L1Loss()

            confidences_loss = confidences_loss_fn(detector_out.confidences, confidences_labels)
            diffs_loss = diffs_loss_fn(detector_out.diffs, diffs_labels)

            total_loss = confidences_loss + diffs_loss

            nms_boxes = []
            for j_batch in range(batch_size):
                pred_boxes = torch.cat((detector_out.pred_bbox_n[j_batch].T, detector_out.pred_bbox_p[j_batch].T)).T

                confidences_batch = detector_out.confidences[j_batch].flatten()
                confidences_batch_p = confidences_batch[detector_out.idx_p[j_batch]]
                confidences_batch_n = confidences_batch[detector_out.idx_n[j_batch]]
                confidences_batch = torch.cat((confidences_batch_n, confidences_batch_p))

                nms_indices = ops.nms(pred_boxes.T, confidences_batch, 0.7)
                nms_boxes_i = pred_boxes[:, nms_indices]

                print(nms_boxes_i.shape)
                nms_boxes.append(utils.tensor_to_labels(nms_boxes_i, model.H, model.W))

                DataManager.plot_num(x_batch[j_batch].view((model.H, model.W)), nms_boxes[j_batch])


def main():
    train_set = MNISTAugDataset(10)
    test_set = MNISTAugDataset(2, test_mode=True)

    model = MnistDetector()

    epochs = 1
    batch_size = 2

    # train(model, train_set, epochs, batch_size, test_set=test_set)
    test(model, test_set)


if __name__ == '__main__':
    main()
