import torch

from lib.mnist_aug.mnist_augmenter import DataManager


# Converts labels (list of dict) to tensors (list of tensors of shape [n, 4])
def labels_to_tensor(labels, H, W):
    # For each bbox, we will have 4 numbers
    tensor = torch.zeros((len(labels), 4), dtype=torch.float32)

    for i in range(len(labels)):
        coordinates = [
            labels[i]["cx"] / W,
            labels[i]["cy"] / H,
            labels[i]["width"] / W,
            labels[i]["height"] / H,
        ]
        tensor[i] = torch.FloatTensor(coordinates)

    return tensor.T


def tensor_to_labels(tensor, H=1, W=1):
    # type: (torch.Tensor, int, int) -> list[dict]
    """
    tensor: shape (4, n) of format (x1y1x2y2)
    """
    return [
        {
            "x1": tensor[0, i] * W,
            "y1": tensor[1, i] * H,
            "x2": tensor[2, i] * W,
            "y2": tensor[3, i] * H,
        }
        for i in range(tensor.shape[1])
    ]


def get_shapes_from_sizes_ratios(sizes, ratios):
    sizes_ = torch.tensor(sizes, dtype=torch.float32).repeat_interleave(
        len(ratios)
    )  # [1, 2, 3] => [1, 1, 2, 2, 3, 3]
    ratios_sqrt = torch.sqrt(torch.tensor(ratios, dtype=torch.float32)).repeat(
        len(sizes)
    )

    w = sizes_ / ratios_sqrt
    h = sizes_ * ratios_sqrt

    return w, h


def get_anchor_centers(W, H):
    hs = (
        torch.arange(1 / H / 2, 1, 1 / H).view((H, 1)).expand((H, W))
    )  # [0, 0.5, 1] => [(0,0,0), (0.5,0.5,0.5), (1,1,1)]
    ws = (
        torch.arange(1 / W / 2, 1, 1 / W).view((1, W)).expand((H, W))
    )  # [0, 0.5, 1] => [(0,0.5,1), (0,0.5,1), (0,0.5,1)]

    return ws, hs  # 2 channels


# Generates a tensor of anchors
def generate_anchors(shape, sizes, ratios):
    k = len(sizes) * len(ratios)
    W, H = shape

    cx, cy = get_anchor_centers(W, H)
    w, h = get_shapes_from_sizes_ratios(sizes, ratios)

    cx = cx.view((H, W, 1)).expand((H, W, k)).flatten()
    cy = cy.view((H, W, 1)).expand((H, W, k)).flatten()

    w = w.view((1, k)).expand((H * W, k)).flatten()
    h = h.view((1, k)).expand((H * W, k)).flatten()

    return torch.stack((cx, cy, w, h))


# TODO: Pass in a batch of bboxes and single set of anchors.
def get_iou_map(boxes1, boxes2):
    n1 = boxes1.shape[1]
    n2 = boxes2.shape[1]

    boxes1 = boxes1.repeat_interleave(n2).reshape(
        (4, n1 * n2)
    )  # [1, 2] => [1, 1, 2, 2]
    boxes2 = boxes2.repeat((1, n1))  # [1, 2] => [1, 2, 1, 2]

    wb = boxes1[2]
    wa = boxes2[2]

    hb = boxes1[3]
    ha = boxes2[3]

    cxb = boxes1[0]
    cxa = boxes2[0]

    cyb = boxes1[1]
    cya = boxes2[1]

    wb_half = wb / 2
    hb_half = hb / 2

    x2b = cxb + wb_half
    x1b = cxb - wb_half

    y2b = cyb + hb_half
    y1b = cyb - hb_half

    wa_half = wa / 2
    ha_half = ha / 2

    x2a = cxa + wa_half
    x1a = cxa - wa_half

    y2a = cya + ha_half
    y1a = cya - ha_half

    x2 = torch.min(torch.stack((x2a, x2b)), 0)[0]
    x1 = torch.max(torch.stack((x1a, x1b)), 0)[0]

    y2 = torch.min(torch.stack((y2a, y2b)), 0)[0]
    y1 = torch.max(torch.stack((y1a, y1b)), 0)[0]

    w = x2 - x1
    h = y2 - y1

    w[w < 0] = 0
    h[h < 0] = 0

    intersection = w * h
    union = (wb * hb) + (wa * ha) - intersection
    iou = intersection / union

    return iou.view((n1, n2))


def raise_bbox_iou(iou: torch.Tensor, threshold: float):
    """
    Raises iou to max(threshold, iou_max_a) for each bbox and max anchor iou
    Parameters
    ----------
    iou: Tensor of shape (n_bbox, k*H*W)
    threshold: threshold to raise iou to

    Returns
    -------
    iou with raised values
    """

    iou_max_a, iou_argmax = torch.max(iou, 1)
    iou_max_a = torch.max(iou_max_a, torch.ones(len(iou_max_a)) * threshold)
    iou[range(len(iou)), tuple(iou_argmax)] = iou_max_a
    return iou


def get_diffs(bboxes, anchors, max_iou, argmax_iou, k, H, W) -> torch.Tensor:
    """
    Parameters
    ----------
    bboxes: Tensor of shape (4, n_bboxes)
    anchors: Tensor of shape (4, H*W*k)
    max_iou: Tensor of shape (H*W*k) max of each anchor
    argmax_iou: Tensor of shape (H*W*k) argmax of each anchor
    k: well, k. The # of anchors each pixel
    H: Height of feature map
    W: Width of feature map

    Returns
    -------
    diffs: A Tensor of shape (4, k, H, W)

    Steps:
    1. Find argmax IOUs
    2. Extract bbox coordinates of shape (4, H*W*k)
    3. Find diffs for each pair
    """

    invalid_indices = torch.nonzero(max_iou == 0).view((-1))

    bboxes_max = bboxes[:, argmax_iou]

    tx = (bboxes_max[0] - anchors[0]) / anchors[2]
    ty = (bboxes_max[1] - anchors[1]) / anchors[3]

    tw = torch.log(bboxes_max[2] / anchors[2])
    th = torch.log(bboxes_max[3] / anchors[3])

    diffs = torch.stack((tx, ty, tw, th))  # Shape: (4, n_anchors)

    len_invalid = len(invalid_indices)
    diffs[:, invalid_indices] = torch.tensor(
        [float("nan") for _ in range(4 * len_invalid)]
    ).view((4, len_invalid))

    return diffs.view((4, k, H, W))


def get_confidences(max_iou, confidence_threshold: float, shape) -> torch.Tensor:
    """
    Parameters
    ----------
    max_iou: Tensor of shape (H*W*k), this is the IOU with best bounding box for each anchor
    confidence_threshold: a real number between 0 to 1. All values >= this will be 1. Rest 0
    shape: shape of output

    Returns
    -------
    confidences: Tensor of shape (k, H, W)
    """

    max_iou = max_iou.clone()
    max_iou[max_iou < confidence_threshold] = 0
    max_iou[max_iou >= confidence_threshold] = 1

    return max_iou.view(shape)


def sample_pn_indices(
    confidences: torch.Tensor, threshold_p: float, threshold_n: float, b_samples: int
):
    """
    Parameters
    ----------
    confidences: A flat 1D tensor of confidences
    threshold_p: something like 0.7
    threshold_n: something like 0.3
    b_samples: b number of samples. Something like 256 / 300

    Returns
    -------
    A tuple of two tensors. Each containing arbitrary number of indices.
    """

    positive_indices = torch.nonzero(confidences >= threshold_p).flatten(0)
    negative_indices = torch.nonzero(confidences <= threshold_n).flatten(0)

    bp = min(len(positive_indices), b_samples // 2)
    sampled_indices = (
        [] if bp <= 0 else torch.multinomial(torch.ones(len(positive_indices)), bp)
    )  # Sampled
    positive_indices = positive_indices[sampled_indices]

    bn = min(len(negative_indices), b_samples - bp)
    sampled_indices = torch.multinomial(
        torch.ones(len(negative_indices)), bn
    )  # Sampled
    negative_indices = negative_indices[sampled_indices]

    return positive_indices, negative_indices


def centers_to_diag(boxes):
    """
    Parameters
    ----------
    boxes: tensor of shape (4, n)

    Returns
    -------
    boxes of shape (4, n)
    """

    cx = boxes[0]
    cy = boxes[1]
    w = boxes[2] / 2
    h = boxes[3] / 2

    x1 = cx - w
    y1 = cy - h
    x2 = cx + w
    y2 = cy + h

    return torch.stack((x1, y1, x2, y2))  # (4, n)


def apply_diff(anchor, diffs):
    cxa = anchor[0]
    cya = anchor[1]
    wa = anchor[2]
    ha = anchor[3]

    cxd = diffs[0]
    cyd = diffs[1]
    wd = diffs[2]
    hd = diffs[3]

    cxb = cxa + cxd * wa
    cyb = cya + cyd * ha
    wb = wa * torch.exp(wd)
    hb = ha * torch.exp(hd)

    return torch.stack((cxb, cyb, wb, hb))


def get_pred_boxes(diffs: torch.Tensor, anchors: torch.Tensor, idx: torch.Tensor):
    """
    Parameters
    ----------
    idx: 1D indices tensor
    diffs: Tensor of shape (4, k, H, W)
    anchors: Tensor of shape (4, k*H*W)

    Returns
    -------
    A tuple of two tensors -> (4, np), (4, nn)

    Steps
    -------
    1. Extract +ve anchors
    2. Flatten out diffs at dim 1 to make a diffs tensor of shape (4, k*H*W)
    3. Extract diffs
    4. Apply diffs to anchors
    5. return pred bboxes
    """

    anchors = anchors[:, idx]

    diffs = diffs.view((4, -1))
    diffs = diffs[:, idx]

    bb = apply_diff(anchors, diffs)

    return bb  # (cx, cy, w, h)


def get_tiny_box_indices(coords, min_side):
    # type: (torch.Tensor, float) -> torch.Tensor
    """
    Parameters
    -------
    coords: shape of (4, n) of format (x1 y1 x2 y2)
    """

    w = coords[2] - coords[0]
    h = coords[3] - coords[1]

    mins = torch.min(w, h)
    return mins > min_side


def main():
    k = 9
    H = 112
    W = 112
    Wp = 14
    Hp = 14
    b_regions = 256

    threshold_p = 0.6
    threshold_n = 0.3

    y = [
        [
            {
                "id": 0,
                "class": 9,
                "class_one_hot": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "x1": 40,
                "y1": 26,
                "x2": 70,
                "y2": 56,
                "cx": 55.0,
                "cy": 41.0,
                "height": 30,
                "width": 30,
            },
            {
                "id": 1,
                "class": 3,
                "class_one_hot": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "x1": 13,
                "y1": 78,
                "x2": 50,
                "y2": 112,
                "cx": 31.5,
                "cy": 95.0,
                "height": 34,
                "width": 37,
                "type": "number",
            },
            {
                "id": 2,
                "class": 6,
                "class_one_hot": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "x1": 78,
                "y1": 35,
                "x2": 112,
                "y2": 70,
                "cx": 95.0,
                "cy": 52.5,
                "height": 35,
                "width": 34,
                "type": "number",
            },
            {
                "id": 3,
                "class": 1,
                "class_one_hot": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "x1": 35,
                "y1": 46,
                "x2": 84,
                "y2": 95,
                "cx": 59.5,
                "cy": 70.5,
                "height": 49,
                "width": 49,
                "type": "number",
            },
            {
                "id": 4,
                "class": 2,
                "class_one_hot": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "x1": 20,
                "y1": 55,
                "x2": 42,
                "y2": 77,
                "cx": 31.0,
                "cy": 66.0,
                "height": 22,
                "width": 22,
                "type": "number",
            },
            {
                "id": 5,
                "class": 1,
                "class_one_hot": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "x1": 60,
                "y1": 11,
                "x2": 95,
                "y2": 46,
                "cx": 77.5,
                "cy": 28.5,
                "height": 35,
                "width": 35,
                "type": "number",
            },
            {
                "id": 6,
                "class": 1,
                "class_one_hot": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "x1": 60,
                "y1": 79,
                "x2": 102,
                "y2": 112,
                "cx": 81.0,
                "cy": 95.5,
                "height": 33,
                "width": 42,
                "type": "number",
            },
            {
                "id": 7,
                "class": 8,
                "class_one_hot": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                "x1": 4,
                "y1": 14,
                "x2": 51,
                "y2": 61,
                "cx": 27.5,
                "cy": 37.5,
                "height": 47,
                "width": 47,
                "type": "number",
            },
        ],
        [
            {
                "id": 0,
                "class": 4,
                "class_one_hot": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "x1": 33,
                "y1": 28,
                "x2": 87,
                "y2": 82,
                "cx": 60.0,
                "cy": 55.0,
                "height": 54,
                "width": 54,
                "type": "number",
            },
            {
                "id": 1,
                "class": 0,
                "class_one_hot": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "x1": 68,
                "y1": 69,
                "x2": 103,
                "y2": 104,
                "cx": 85.5,
                "cy": 86.5,
                "height": 35,
                "width": 35,
                "type": "number",
            },
            {
                "id": 2,
                "class": 1,
                "class_one_hot": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "x1": 62,
                "y1": 0,
                "x2": 95,
                "y2": 33,
                "cx": 78.5,
                "cy": 16.5,
                "height": 33,
                "width": 33,
                "type": "number",
            },
            {
                "id": 3,
                "class": 1,
                "class_one_hot": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "x1": 16,
                "y1": 88,
                "x2": 48,
                "y2": 112,
                "cx": 32.0,
                "cy": 100.0,
                "height": 24,
                "width": 32,
                "type": "number",
            },
        ],
    ]
    DataManager.plot_num(torch.ones((H, W)), y[1])

    y_ = labels_to_tensor(
        y[1], H, W
    )  # Tensor of shape (4, n) -> (cx, cy, w, h) normalized
    anchors_tensor = generate_anchors(
        shape=(Wp, Hp), sizes=(0.15, 0.45, 0.75), ratios=(0.5, 1, 2)
    )  # Tensor of shape (4, k*H*W) -> cy, cy, w, h

    # Looped
    i = 0
    iou = get_iou_map(y_[i], anchors_tensor)
    iou = raise_bbox_iou(iou, threshold_p)
    iou_max, iou_argmax = torch.max(iou, 0)  # Shape (k*H*W)

    confidences = get_confidences(iou_max, threshold_p, (k, Hp, Wp))
    diffs = get_diffs(y_[i], anchors_tensor, iou_max, iou_argmax, k, Hp, Wp)

    idx_p, idx_n = sample_pn_indices(iou_max, threshold_p, threshold_n, b_regions)

    diffs_pred = diffs
    pred_bbox_p, pred_bbox_n = get_pred_boxes(
        diffs_pred, anchors_tensor, idx_p, idx_n
    )  # (4, n) (cx, cy, w, h)
    pred_bbox_p = centers_to_diag(pred_bbox_p)
    pred_bbox_n = centers_to_diag(pred_bbox_n)


if __name__ == "__main__":
    main()
