from lib.transporter import Transporter
import json


class WebsocketTransporter(Transporter):
    def __init__(self, project_id, trace_id=None, consumer=None):
        super().__init__(project_id, trace_id)
        self.consumer = consumer

    def send(self, data):
        self.set_trace_id()
        data['trace_id'] = self.trace_id
        self.consumer.send(text_data=json.dumps(data))

