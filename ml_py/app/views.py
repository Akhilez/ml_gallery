from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from app.nlp.next_char.next_char import NextChar


class PreLoaded:
    next_char = NextChar()


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
