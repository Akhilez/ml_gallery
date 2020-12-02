import torch

k = 9
W = 14
H = 14
B = 256
b = 2


# Converts labels (list of dict) to tensors (list of tensors of shape [n, 4])
def labels_to_tensor(labels):
    tensors = []
    for labels_ in labels:

        # For each bbox, we will have 4 numbers
        tensor = torch.zeros((len(labels_), 4), dtype=torch.float32)

        for i in range(len(labels_)):
            coordinates = [
                labels_[i]['cx'] / W,
                labels_[i]['cy'] / H,
                labels_[i]['width'] / W,
                labels_[i]['height'] / H,
            ]
            tensor[i] = torch.FloatTensor(coordinates)

        tensors.append(tensor)

    return tensors


def get_shapes_from_sizes_ratios(sizes, ratios):
    sizes_ = torch.tensor(sizes, dtype=torch.float32).repeat_interleave(len(ratios))  # [1, 2, 3] => [1, 1, 2, 2, 3, 3]
    ratios_sqrt = torch.sqrt(torch.tensor(ratios, dtype=torch.float32)).repeat(len(sizes))

    w = sizes_ / ratios_sqrt
    h = sizes_ * ratios_sqrt

    return w, h


def get_anchor_centers(W, H):
    hs = torch.arange(1/H/2, 1, 1/H).view((H, 1)).expand((H, W))  # [0, 0.5, 1] => [(0,0,0), (0.5,0.5,0.5), (1,1,1)]
    ws = torch.arange(1/W/2, 1, 1/W).view((1, W)).expand((H, W))  # [0, 0.5, 1] => [(0,0.5,1), (0,0.5,1), (0,0.5,1)]

    return ws, hs  # 2 channels


# Generates a tensor of anchors
def generate_anchors(shape, sizes, ratios):
    k = len(sizes) * len(ratios)
    W, H = shape

    cx, cy = get_anchor_centers(W, H)
    w, h = get_shapes_from_sizes_ratios(sizes, ratios)

    cx = cx.view((H, W, 1)).expand((H, W, k)).flatten()
    cy = cy.view((H, W, 1)).expand((H, W, k)).flatten()

    w = w.view((1, -1)).expand((H * W, -1)).flatten()
    h = h.view((1, -1)).expand((H * W, -1)).flatten()

    return torch.stack((cx, cy, w, h)).T


y = [
    [
        {'id': 0,
         'class': 9,
         'class_one_hot': [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
         'x1': 40,
         'y1': 26,
         'x2': 70,
         'y2': 56,
         'cx': 55.0,
         'cy': 41.0,
         'height': 30,
         'width': 30,
         'type': 'number'},
        {'id': 1,
         'class': 3,
         'class_one_hot': [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
         'x1': 13,
         'y1': 78,
         'x2': 50,
         'y2': 112,
         'cx': 31.5,
         'cy': 95.0,
         'height': 34,
         'width': 37,
         'type': 'number'},
        {'id': 2,
         'class': 6,
         'class_one_hot': [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
         'x1': 78,
         'y1': 35,
         'x2': 112,
         'y2': 70,
         'cx': 95.0,
         'cy': 52.5,
         'height': 35,
         'width': 34,
         'type': 'number'},
        {'id': 3,
         'class': 1,
         'class_one_hot': [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
         'x1': 35,
         'y1': 46,
         'x2': 84,
         'y2': 95,
         'cx': 59.5,
         'cy': 70.5,
         'height': 49,
         'width': 49,
         'type': 'number'},
        {'id': 4,
         'class': 2,
         'class_one_hot': [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
         'x1': 20,
         'y1': 55,
         'x2': 42,
         'y2': 77,
         'cx': 31.0,
         'cy': 66.0,
         'height': 22,
         'width': 22,
         'type': 'number'},
        {'id': 5,
         'class': 1,
         'class_one_hot': [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
         'x1': 60,
         'y1': 11,
         'x2': 95,
         'y2': 46,
         'cx': 77.5,
         'cy': 28.5,
         'height': 35,
         'width': 35,
         'type': 'number'},
        {'id': 6,
         'class': 1,
         'class_one_hot': [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
         'x1': 60,
         'y1': 79,
         'x2': 102,
         'y2': 112,
         'cx': 81.0,
         'cy': 95.5,
         'height': 33,
         'width': 42,
         'type': 'number'},
        {'id': 7,
         'class': 8,
         'class_one_hot': [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
         'x1': 4,
         'y1': 14,
         'x2': 51,
         'y2': 61,
         'cx': 27.5,
         'cy': 37.5,
         'height': 47,
         'width': 47,
         'type': 'number'}],
    [
        {'id': 0,
         'class': 4,
         'class_one_hot': [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
         'x1': 33,
         'y1': 28,
         'x2': 87,
         'y2': 82,
         'cx': 60.0,
         'cy': 55.0,
         'height': 54,
         'width': 54,
         'type': 'number'},
        {'id': 1,
         'class': 0,
         'class_one_hot': [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         'x1': 68,
         'y1': 69,
         'x2': 103,
         'y2': 104,
         'cx': 85.5,
         'cy': 86.5,
         'height': 35,
         'width': 35,
         'type': 'number'},
        {'id': 2,
         'class': 1,
         'class_one_hot': [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
         'x1': 62,
         'y1': 0,
         'x2': 95,
         'y2': 33,
         'cx': 78.5,
         'cy': 16.5,
         'height': 33,
         'width': 33,
         'type': 'number'},
        {'id': 3,
         'class': 1,
         'class_one_hot': [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
         'x1': 16,
         'y1': 88,
         'x2': 48,
         'y2': 112,
         'cx': 32.0,
         'cy': 100.0,
         'height': 24,
         'width': 32,
         'type': 'number'}]]

y_ = labels_to_tensor(y)

anchors_tensor = generate_anchors(shape=(W, H), sizes=(15, 45, 75), ratios=(0.5, 1, 2))

print(anchors_tensor.shape)
print([y_i.shape for y_i in y_])

