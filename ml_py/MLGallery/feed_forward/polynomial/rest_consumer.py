from django.http import HttpResponse
from MLGallery.feed_forward.polynomial.trainer import PolyRegTrainer
from lib.job_handler import JobHandler
import json


def receive(request):
    if request.method == "POST":
        body = json.loads(request.body)
        trainer = PolyRegTrainer()
        job_handler = JobHandler(trainer, 'learn_curve', request.session, send_callback=send)
        return job_handler.receive(body)
    return send({})


def send(data):
    return HttpResponse(json.dumps(data))

