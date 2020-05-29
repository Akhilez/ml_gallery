import json

from MLGallery.feed_forward.polynomial.trainer import PolyRegTrainer


class CommonConsumer:
    def __init__(self, transporter):
        self.trainer = None
        self.transporter = transporter

    def init_trainer(self):
        """
        1. Initialize order, x, y, w, b,
        2. Send sample data to client.
        """

        self.trainer = PolyRegTrainer(self)
        self.trainer.x, self.trainer.y = self.trainer.get_random_sample_data(50)

        return self.transporter.send(data={
            'action': 'init',
            'data': self.trainer.get_float_data(),
        })


