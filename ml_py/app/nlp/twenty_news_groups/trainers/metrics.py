import numpy as np
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import categorical_crossentropy
import math


class Metrics:
    def __init__(self):
        self.n_classes = 20

        self.losses = []
        self.accuracies = []
        self.f1macros = []

        self._epoch_loss = 0
        self._epoch_accuracy = 0
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

        self.n_batches = 0
        self.n_epochs = 0

    def record_batch(self, loss, yh, y):
        self.n_batches += 1
        self._epoch_loss += loss
        self._epoch_accuracy += self.find_accuracy(yh, y)
        self.confusion_matrix += self.get_confusion_matrix(yh, y, self.n_classes)

    def record_epoch(self):
        self.losses.append(self._epoch_loss / self.n_batches)
        self.accuracies.append(self._epoch_accuracy / self.n_batches)
        self.f1macros.append(self.get_f1_macro(self.confusion_matrix))

        self.n_epochs += 1
        self._epoch_loss = 0
        self._epoch_accuracy = 0
        self.n_batches = 0

    @staticmethod
    def get_f1_macro(confusion_matrix):
        n_classes = len(confusion_matrix)

        tp = np.zeros((n_classes,))
        fn = np.zeros((n_classes,))
        fp = np.zeros((n_classes,))

        for i_real in range(n_classes):
            for i_pred in range(n_classes):
                value = confusion_matrix[i_real][i_pred]
                if i_real == i_pred:
                    tp[i_real] = value
                else:
                    fn[i_real] += value
                    fp[i_pred] += value

        fn[fn == 0] = 1.0e-06
        fp[fp == 0] = 1.0e-06

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)

        p_macro = sum(precision) / n_classes
        r_macro = sum(recall) / n_classes

        f_macro = 2 * p_macro * r_macro / (p_macro + r_macro)
        return f_macro

    @staticmethod
    def get_confusion_matrix(yh, y, n_classes):
        max_yh = yh.argmax(axis=1)
        max_y = y.argmax(axis=1)

        conf = np.zeros((n_classes, n_classes))
        for i in range(len(y)):
            conf[max_y[i]][max_yh[i]] += 1
        return conf

    @staticmethod
    def find_accuracy(y_hat, y_real):
        return sum(np.argmax(y_hat, axis=1) == np.argmax(y_real, axis=1)) / len(y_hat)


class MetricsCallback(Callback):
    def __init__(self, metrics_obj, data_gen, steps):
        super(MetricsCallback, self).__init__()
        self.metrics = metrics_obj
        self.data_gen = data_gen
        self.steps = steps

    def on_epoch_end(self, epoch, logs=None):
        for batch_i in range(math.ceil(self.steps)):
            x_batch, y_batch = next(self.data_gen)

            y_hat = self.model.predict(x_batch)

            loss = categorical_crossentropy(y_batch, y_hat)

            self.metrics.record_batch(float(sum(loss)), y_hat, y_batch)

        self.metrics.record_epoch()

        print(f"\tval_f1_macro: {self.metrics.f1macros[-1]}")


class MetricsWrapper(metrics.Metric):
    def __init__(self, metrics_object, name="f1_macro", **kwargs):
        super(MetricsWrapper, self).__init__(name=name, **kwargs)
        self.obj = metrics_object

    def update_state(self, y_true, y_pred, **kwargs):
        pass

    def result(self):
        if len(self.obj.f1macros) > 0:
            return self.obj.f1macros[-1]
        return 0
