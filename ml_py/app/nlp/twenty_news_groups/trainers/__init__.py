from abc import ABC, abstractmethod


class Trainer(ABC):
    @abstractmethod
    def train(self, epochs, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def save(self, path):
        pass
