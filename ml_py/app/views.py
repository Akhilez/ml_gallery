from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from app.nlp.next_char.next_char import NextChar
from app.vision.positional_mnist.positional_mnist import PositionalCNN


class PreLoaded:
    next_char = NextChar()
    positional_cnn = PositionalCNN()


def home(request):
    return render(request, 'app/home.html', {'name': 'Akhil'})


def selective_generator_page(request):
    return HttpResponse("MNIST")


@csrf_exempt
def rest_learn_curve(request):
    from app.feed_forward.polynomial import rest_consumer
    return rest_consumer.receive(request)


@api_view(['GET'])
def next_char(request):
    text = request.GET.get('text')
    if text is None:
        return Response({}, status=status.HTTP_400_BAD_REQUEST)
    return Response({'pred': PreLoaded.next_char.predict(text)}, status=status.HTTP_200_OK)


@api_view(['GET'])
def positional_cnn(request):
    image = request.GET.get('image')
    if image is None:
        return Response({}, status=status.HTTP_400_BAD_REQUEST)
    cls, pos = PreLoaded.positional_cnn.predict(image)
    return Response({'class': cls, 'position': pos}, status=status.HTTP_200_OK)
