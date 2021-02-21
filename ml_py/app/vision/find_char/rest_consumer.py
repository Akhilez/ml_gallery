from django.http import HttpResponse
from lib.job_handler import JobHandler, all_jobs
import json

project_id = "find_char"


def receive(request):
    if request.method == "POST":
        body = json.loads(request.body)
        trainer = None

        job_handler = None
        if body.get("job_id") is not None:
            job_handler = all_jobs.get(body.get("job_id"))
        if job_handler is None:
            job_handler = JobHandler(trainer, "learn_curve", send_callback=send)

        return job_handler.receive(body)
    return send({})


def send(data):
    return HttpResponse(json.dumps(data))
