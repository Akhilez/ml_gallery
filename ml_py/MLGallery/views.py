from django.http import HttpResponse
from django.shortcuts import render


def home(request):
    return HttpResponse("Home")


def selective_generator_page(request):
    return HttpResponse("MNIST")


def rest_learn_curve(request):
    from MLGallery.feed_forward.polynomial import view_handler
    return view_handler.get_view(request)
