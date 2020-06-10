import uuid
from ml_py.settings import logger
import threading


class JobHandler:

    def __init__(self, trainer, job_title, session, send_callback):
        super().__init__()
        self.title = job_title
        self.session = session
        self.send = send_callback
        self.trainer = trainer

    def init_session(self):
        self.session['job_id'] = str(uuid.uuid1())
        self.trainer.x, self.trainer.y = self.trainer.get_random_sample_data(50)
        return self.send(data={
            'action': 'init',
            'job_id': self.session['job_id'],
            'data': self.trainer.get_float_data(),
        })

    def receive(self, data: dict):
        """
        Parameters
        ----------
        data: dict = {
            action: 'init' | ...,
            job_id: 'AKSJD#2304',
            data: some data
        }
        """
        if data['action'] == 'init':
            return self.init_session()

        if self.trace_id is None:
            logger.error("No trace ID found")
            return

        action = data['action']

        if action == 'start_training':
            threading.Thread(target=self.trainer.start_training).start()

        if action == 'stop_training':
            self.trainer.stop_training()

        if action == 'change_order':
            threading.Thread(target=self.trainer.change_order, args=(data['order'],)).start()

        if action == 'new_point':
            self.trainer.add_new_point(data['x'], data['y'])

        if action == 'clear_data':
            self.trainer.clear_data()
