import time
import json
from channels.generic.websocket import WebsocketConsumer


class ChatConsumer(WebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_id = None  # Manage the trace using a DB or some redis queue or whatever that is stateful
        # Maintain a list of trace ids
        self.last_message = None

    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data=None, bytes_data=None):
        text_data_json = json.loads(text_data)

        if text_data_json['type'] == 'sendMessage':
            if self.trace_id is None:
                self.trace_id = text_data_json['traceId']

            message = text_data_json['message']
            self.last_message = message

            self.send(text_data=json.dumps({
                'message': message
            }))
        elif text_data_json['type'] == 'action':
            if self.trace_id is not None:
                if self.last_message is not None:
                    self.send(text_data=json.dumps({'message': self.last_message[::-1]}))

        time.sleep(5)

        self.send(text_data=json.dumps({
            'message': 'This is sent after 5 seconds.'
        }))
