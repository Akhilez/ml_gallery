import torch
from lib.nn_utils import get_scaled_random_weights


class PolyRegTrainer(torch.nn.Module):

    def __init__(self, consumer):
        super().__init__()
        self.w = None
        self.b = None
        self.consumer = consumer
        self.order = 5
        self.epochs = 20000
        self.update_interval = 100
        self.optimizer = None
        self.must_train = False

        # -------- Current Epoch -----------
        self.epoch = 0
        self.loss = 0

    def init_weights(self):
        self.w = get_scaled_random_weights(self.order)
        self.b = torch.zeros(1)
        self.optimizer = torch.optim.Adam([self.w, self.b])

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

    def start_training(self, data):
        """
        1. Initialize model with weights
        Parameters
        ----------
        data: Iterable of shape (any, 2)

        Returns
        -------

        """
        data = torch.stack(data)
        x = data[:, 0]
        y = data[:, 1]

        self.init_weights()
        self.must_train = True

        for epoch in range(self.epochs):

            yh = self(x)
            self.loss = sum((y - yh) ** 2)

            self.loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.epoch = epoch

            if epoch % self.update_interval == 0:
                self.consumer.send_status()

            if not self.must_train:
                return

    def stop_training(self):
        self.must_train = False
