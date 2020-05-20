import torch
from lib.nn_utils import get_scaled_random_weights
from ml_py.settings import logger


class PolyRegTrainer(torch.nn.Module):

    def __init__(self, consumer):
        super().__init__()
        self.order = 5
        self.w = get_scaled_random_weights([self.order])
        self.b = torch.zeros(1, requires_grad=True)
        self.consumer = consumer
        self.epochs = 20000
        self.update_interval = 1000
        self.optimizer = torch.optim.Adam([self.w, self.b])
        self.must_train = False
        self.x = None
        self.y = None

        # -------- Current Epoch -----------
        self.epoch = 0
        self.loss = 0

    def forward(self, x):
        """
        Parameters
        ----------
        x: tensor of shape (batch, 1)

        Returns
        -------
        yh: the output of the neuron
        """
        x = torch.stack([x ** i for i in range(self.order, 0, -1)])
        return sum((x.T * self.w).T) + self.b

    def start_training(self):
        """
        Parameters
        ----------
        data: Iterable of shape (any, 2)

        Returns
        -------

        """

        self.must_train = True

        for epoch in range(self.epochs + 1):

            yh = self(self.x)
            loss = sum((self.y - yh) ** 2)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.epoch = epoch
            self.loss = float(loss)

            if not self.must_train:
                return self.consumer.send_update_status()

            if epoch % self.update_interval == 0:
                self.consumer.send_update_status()

    def stop_training(self):
        self.must_train = False

    def change_order(self, new_order):
        pass  # TODO: Change order and its dependencies

    def get_float_parameters(self):
        w = self.w.tolist()
        w.extend(self.b.tolist())
        return w

    def get_float_data(self):
        return [self.x.tolist(), self.y.tolist()]

    def add_new_point(self, x, y):
        if self.x is None:
            self.x = torch.tensor([x])
            self.y = torch.tensor([y])
        else:
            self.x = torch.cat((self.x, torch.tensor([x])))
            self.y = torch.cat((self.y, torch.tensor([y])))

    def get_random_sample_data(self, size: int):
        """
        1. x = random from -1 to 1
        2. w = random from -0.01 to 0.01
        3. y = wx + b
        4. return x, y
        """

        x = torch.FloatTensor(size).uniform_(-1, 1)
        w = torch.FloatTensor(self.order).uniform_(-0.01, 0.01)

        new_x = torch.stack([x ** i for i in range(self.order, 0, -1)])
        y = sum((new_x.T * w).T)

        return x, y

    def clear_data(self):
        self.x = None
        self.y = None
