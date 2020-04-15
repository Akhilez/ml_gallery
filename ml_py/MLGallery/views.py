from django.http import HttpResponse
from django.shortcuts import render


def home(request):
    return HttpResponse("Home")


def selective_generator_page(request):

    return HttpResponse("MNIST")
