import time
import torch
import threading
from ml_py.settings import logger


class PolyRegTrainer(torch.nn.Module):

    def __init__(self, consumer=None):
        super().__init__()
        self.order = 5
        self.w = None
        self.b = None
        self.consumer = consumer
        self.epochs = 5000
        self.update_interval = 500
        self.optimizer = None
        self.must_train = False
        self.x = None
        self.y = None

        self.init_weights()

        # -------- Current Epoch -----------
        self.epoch = 0
        self.loss = 0

    def init_weights(self):
        self.w = torch.zeros(self.order, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.w, self.b])

    def forward(self, x):
        x = torch.stack([x ** i for i in range(self.order, 0, -1)])
        return sum((x.T * self.w).T) + self.b

    def start_training(self):
        try:
            logger.info("Starting training.")

            self.must_train = True

            for epoch in range(self.epochs + 1):

                yh = self(self.x)
                loss = sum((self.y - yh) ** 2)

                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.epoch = epoch
                self.loss = float(loss)

                if not self.must_train:
                    logger.info("Stopping training.")
                    return self.update_consumer()

                if epoch % self.update_interval == 0:
                    self.update_consumer()

        except Exception as e:
            logger.exception(e)

        self.must_train = False

    def stop_training(self):
        self.must_train = False

    def update_consumer(self):
        if self.consumer is not None:
            threading.Thread(target=self.consumer.send_update_status, args=(self.get_status_data(),)).start()

    def get_status_data(self):
        return {
            'epoch': self.epoch,
            'train_error': float(self.loss),
            'weights': self.get_float_parameters(),
            'is_training': self.must_train,
        }

    def change_order(self, new_order):
        prev_training = self.must_train
        if prev_training:
            self.stop_training()
        time.sleep(0.5)
        self.order = new_order
        self.init_weights()
        if prev_training:
            self.start_training()

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
            self.x = torch.cat((self.x, torch.tensor([x], dtype=torch.float32)))
            self.y = torch.cat((self.y, torch.tensor([y], dtype=torch.float32)))

    def get_random_sample_data(self, size: int):
        x = torch.FloatTensor(size).uniform_(-1, 1)
        w = torch.tensor([-0.85, -1.6, 2.3, 2.6, -1.2])
        b = torch.tensor([-0.7])

        new_x = torch.stack([x ** i for i in range(self.order, 0, -1)])
        y = sum((new_x.T * w).T) + b

        return x, y

    def clear_data(self):
        self.x = None
        self.y = None
        self.init_weights()
