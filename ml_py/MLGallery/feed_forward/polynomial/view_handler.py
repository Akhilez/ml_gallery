from django.http import HttpResponse
from ml_py.settings import logger
from MLGallery.feed_forward.polynomial.common_consumer import CommonConsumer
from lib.transporter.rest_transporter import RestTransporter
import json


def get_view(request):
    if request.method == "POST":
        body = json.loads(request.body)
        trace_id = body.get('trace_id')
        common_consumer = CommonConsumer(RestTransporter('learn_curve', trace_id))
        action = body.get('action')
        logger.info(f'Action: {action}. {request.body=}')
        if action is not None:
            if action == 'init':
                return common_consumer.init_trainer()

    return HttpResponse(json.dumps({}))
