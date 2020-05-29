from lib.transporter import Transporter

import json
from django.http import HttpResponse


class RestTransporter(Transporter):
    def __init__(self, project_id, trace_id=None):
        super().__init__(project_id, trace_id)

    def send(self, data):
        self.set_trace_id()
        data['trace_id'] = self.trace_id
        return HttpResponse(json.dumps(data))

