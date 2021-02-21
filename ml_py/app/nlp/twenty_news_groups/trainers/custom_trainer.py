from trainers.batch_generator import BatchGenerator
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
import matplotlib.pyplot as plt
from trainers import Trainer
from trainers.metrics import MetricsCallback, Metrics


class CustomTrainer(Trainer):
    def __init__(self, load_path=None):
        self.vocab_size = 10000
        self.embed_size = 128

        self.metrics = Metrics()

        self.model = self._get_model(load_path)

        self.train_batch_size = 30
        self.test_batch_size = 30

        self.train_size = 11083
        self.test_size = 7761

    def train(self, epochs, **kwargs):
        train_steps = self.train_size / self.train_batch_size
        test_steps = self.test_size / self.test_batch_size

        train_gen = BatchGenerator(
            data_path="../data/train_custom.csv", batch_size=self.train_batch_size
        ).get_batch_gen()
        test_gen = BatchGenerator(
            data_path="../data/test_custom.csv", batch_size=self.test_batch_size
        ).get_batch_gen()

        return self.model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=test_gen,
            validation_steps=test_steps,
            callbacks=[MetricsCallback(self.metrics, test_gen, test_steps)],
        )

    def _get_model(self, load_path):
        if load_path:
            return tf.keras.models.load_model(load_path)

        model = tf.keras.Sequential(
            [
                Embedding(self.vocab_size, self.embed_size, input_length=150),
                GlobalAveragePooling1D(),
                Dense(128, activation="relu"),
                Dense(20, activation="softmax"),
            ]
        )
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    def predict(self, data):
        return self.model.predict(data)

    def save(self, path):
        self.model.save(path)

    @staticmethod
    def plot_graphs(history, metrics):
        CustomTrainer._plot_graphs(history, "accuracy", metrics)
        CustomTrainer._plot_graphs(history, "loss")

    @staticmethod
    def _plot_graphs(history, string, metrics=None):
        legend = [string, "val_" + string]
        plt.plot(history.history[string])
        plt.plot(history.history["val_" + string])
        if string == "accuracy":
            plt.plot(metrics.f1macros)
            legend.append("f1_macro")
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend(legend)
        plt.show()


if __name__ == "__main__":
    model_path = "models/base.h5"
    trainer = CustomTrainer()  # load_path=model_path)

    # trainer.train_size = 500
    # trainer.test_size = 500

    logs = trainer.train(25)
    trainer.plot_graphs(logs, trainer.metrics)

    trainer.save(model_path)
