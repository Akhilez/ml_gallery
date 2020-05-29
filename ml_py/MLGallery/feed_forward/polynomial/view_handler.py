from django.http import HttpResponse
import json


def get_view(request):
    return HttpResponse("Hii")
