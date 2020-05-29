from abc import ABC, abstractmethod

from lib.trace_manager import TraceManager


class Transporter(ABC):
    def __init__(self, project_id, trace_id=None):
        self.project_id = project_id
        self.trace_id = trace_id

    @abstractmethod
    def send(self, data):
        pass

    def set_trace_id(self):
        if self.trace_id is None:
            import uuid
            self.trace_id = str(uuid.uuid1())
            TraceManager.jobs[self.trace_id] = self
