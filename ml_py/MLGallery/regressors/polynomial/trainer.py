import asyncio

import torch
from lib.nn_utils import get_scaled_random_weights
from ml_py.settings import logger


class PolyRegTrainer(torch.nn.Module):

    def __init__(self, consumer):
        super().__init__()
        self.w = None
        self.b = None
        self.consumer = consumer
        self.order = 5
        self.epochs = 20000
        self.update_interval = 1000
        self.optimizer = None
        self.must_train = False

        # -------- Current Epoch -----------
        self.epoch = 0
        self.loss = 0

    def get_parameters(self):
        return [self.w, self.b]

    def get_float_parameters(self):
        w = self.w.tolist()
        w.extend(self.b.tolist())
        return w

    def init_weights(self):
        self.w = get_scaled_random_weights([self.order])
        self.b = torch.zeros(1)
        self.optimizer = torch.optim.Adam(self.get_parameters())

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

    async def start_training(self, data):
        """
        1. Initialize model with weights
        Parameters
        ----------
        data: Iterable of shape (any, 2)

        Returns
        -------

        """
        data = torch.tensor(data)

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
                logger.info(f'must train: {self.must_train}. epoch: {epoch}')
                asyncio.create_task(self.consumer.send_update_status())

            if not self.must_train:
                return

    def stop_training(self):
        self.must_train = False
