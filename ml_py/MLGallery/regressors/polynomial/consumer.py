import uuid

from channels.generic.websocket import WebsocketConsumer
import json

from MLGallery.regressors.polynomial.trainer import PolyRegTrainer
from lib.trace_manager import TraceManager
from ml_py.settings import logger


class PolyRegConsumer(WebsocketConsumer):
    """
    Receives a json of the following type:
    {
        action: start_training | stop_training,
        trace_id: UUID | None
        data: data
    }

    Sends json:
    {
        action: status_update
        trace_id: UUID
        data: {
            epoch: 10
            train_error: 0.1
            weights: {
                L1: [1.2, -0.3, 4.5]
            }
        }
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_id = None
        self.trainer = None

    def connect(self):
        self.trainer = PolyRegTrainer(self)
        self.accept()

    def disconnect(self, close_code):
        self.trainer.stop_training()
        if self.trace_id is not None:
            del TraceManager.jobs[self.trace_id]

    def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data)

        if self.trace_id is None:
            self.trace_id = str(uuid.uuid1())
            TraceManager.jobs[self.trace_id] = self

        action = data['action']

        if action == 'start_training':
            self.start_training(data)

        if action == 'stop_training':
            logger.info(f'must train in consumer: {self.trainer.must_train}')
            self.trainer.stop_training()

    def start_training(self, data):
        logger.info(f'{data=}')
        import threading
        threading.Thread(target=self.trainer.start_training, args=(data['data'],)).start()

    def send_update_status(self):
        data = {
            'action': 'status_update',
            'trace_id': self.trace_id,
            'data': {
                'epoch': self.trainer.epoch,
                'train_error': float(self.trainer.loss),
                'weights': self.trainer.get_float_parameters()
            }
        }
        self.send(text_data=json.dumps(data))
