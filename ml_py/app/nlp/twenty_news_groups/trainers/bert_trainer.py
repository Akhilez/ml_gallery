import torch
from pytorch_pretrained_bert import BertModel
from trainers.batch_generator import BatchGenerator
from torch import nn
import math
import matplotlib.pyplot as plt
from trainers.metrics import Metrics
from trainers import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertEmailClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertEmailClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 20)

    def forward(self, x, masks=None):
        _, x = self.bert(x, attention_mask=masks, output_all_encoded_layers=False)
        x = self.dropout(x)
        x = nn.functional.softmax(self.linear(x), dim=1)
        return x


class BertTrainer(Trainer):
    def __init__(self, load_path=None):
        self.model = self._get_model(load_path)

        self.train_batch_size = 3
        self.test_batch_size = 3

        self.train_size = 11083
        self.test_size = 7761

    @staticmethod
    def _get_model(load_path):
        if load_path:
            return torch.load(load_path, map_location=device)
        return BertEmailClassifier().to(device)

    def train(self, epochs, train_metrics=None, test_metrics=None):
        train_steps = self.train_size / self.train_batch_size
        test_steps = self.test_size / self.test_batch_size

        train_gen = BatchGenerator(
            data_path="../data/train_bert.csv", batch_size=self.train_batch_size
        ).get_batch_gen()
        test_gen = BatchGenerator(
            data_path="../data/test_bert.csv", batch_size=self.test_batch_size
        ).get_batch_gen()

        optim = torch.optim.Adam(self.model.parameters(), lr=3e-6)

        for epoch in range(epochs):
            self.run_epoch(
                data_gen=train_gen,
                steps=train_steps,
                optim=optim,
                metrics=train_metrics,
            )

            self.run_epoch(
                data_gen=test_gen, steps=test_steps, train=False, metrics=test_metrics
            )

            self.print_progress(epoch, train_metrics, test_metrics)

    def run_epoch(
        self, steps=None, data_gen=None, train=True, optim=None, metrics=None
    ):

        if train:
            self.model.train()
        else:
            self.model.eval()

        for batch_i in range(math.ceil(steps)):

            x_batch, y_batch = next(data_gen)
            y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)
            x_batch_masks = self.create_masks(x_batch)
            x_batch = torch.tensor(x_batch).to(device)

            if train:
                optim.zero_grad()

            y_hat = self.model(x_batch, x_batch_masks)

            loss = nn.functional.binary_cross_entropy(y_hat, y_batch)

            if train:
                loss.backward()
                optim.step()

            metrics.record_batch(
                loss.item(), y_hat.detach().numpy(), y_batch.detach().numpy()
            )

            if train and (batch_i + 1) % (int(steps / 5) + 1) == 0:
                print(
                    f"Epoch: {metrics.n_epochs}\tBatch:{batch_i}\tLoss: {loss.item()}"
                )

        metrics.record_epoch()

    def predict(self, data):
        masks = self.create_masks(data)
        data = torch.tensor(data).to(device)
        with torch.no_grad():
            return self.model(data, masks).to("cpu")

    def save(self, path):
        torch.save(self.model, path)

    @staticmethod
    def print_progress(epoch, train_metrics, test_metrics):
        print(f"Epoch: {epoch + 1}", end="\t")
        if train_metrics:
            print(f"train_loss={train_metrics.losses[-1]}", end="\t")
            print(f"train_accuracy={train_metrics.accuracies[-1]}", end="\t")
            print(f"train_f1={train_metrics.f1macros[-1]}", end="\t")
        if test_metrics:
            print(f"val_loss={test_metrics.losses[-1]}", end="\t")
            print(f"val_accuracy={test_metrics.accuracies[-1]}", end="\t")
            print(f"val_f1={test_metrics.f1macros[-1]}", end="\t")
        print()

    @staticmethod
    def create_masks(x):
        return torch.tensor([[float(i > 0) for i in ii] for ii in x]).to(device)

    @staticmethod
    def clear_gpu_cache():
        if device == "cuda":
            print(str(torch.cuda.memory_allocated(device) / 1000000) + "M")
            torch.cuda.empty_cache()
            print(str(torch.cuda.memory_allocated(device) / 1000000) + "M")

    @staticmethod
    def plot_graphs(train_metrics, test_metrics):
        plt.plot(train_metrics.accuracies)
        plt.plot(test_metrics.accuracies)
        plt.xlabel("Epochs")
        plt.ylabel("accuracy")
        plt.legend(["accuracy", "val_accuracy"])
        plt.show()

        plt.plot(train_metrics.losses)
        plt.plot(test_metrics.losses)
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        plt.legend(["loss", "val_loss"])
        plt.show()

        plt.plot(train_metrics.f1macros)
        plt.plot(test_metrics.f1macros)
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        plt.legend(["f1 macro", "val f1 macro"])
        plt.show()


if __name__ == "__main__":
    model_path = "models/bert_clf.pt"
    trainer = BertTrainer()  # load_path=model_path)

    train_metric = Metrics()
    test_metric = Metrics()

    # trainer.train_size = 10
    # trainer.test_size = 10

    trainer.train(25, train_metric, test_metric)
    trainer.plot_graphs(train_metric, test_metric)

    trainer.save(model_path)
