from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt


def home(request):
    return render(request, 'app/home.html', {'name': 'Akhil'})


def selective_generator_page(request):
    return HttpResponse("MNIST")


@csrf_exempt
def rest_learn_curve(request):
    from app.feed_forward.polynomial import rest_consumer
    return rest_consumer.receive(request)
