from channels.generic.websocket import WebsocketConsumer
import json
from app.feed_forward.polynomial.trainer import PolyRegTrainer
from lib.job_handler import JobHandler


class PolyRegConsumer(WebsocketConsumer):
    """
    Receives a json of the following type:
    {
        action: start_training | stop_training,
        job_id: UUID | None
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
        self.trainer = PolyRegTrainer()
        self.job_handler = JobHandler(
            self.trainer, "learn_curve", self.scope["session"], self.send_callback
        )

    def connect(self):
        self.accept()
        self.job_handler.init_session()

    def disconnect(self, close_code):
        self.trainer.stop_training()
        #  TODO: del all_jobs[self]

    def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data)
        self.job_handler.receive(data)

    def send_callback(self, data):
        self.send(text_data=json.dumps(data))

    def send_update_status(self, data):
        data = {"action": "status_update", "data": data}
        self.send(text_data=json.dumps(data))
