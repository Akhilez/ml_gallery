from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt


def home(request):
    return HttpResponse("Home")


def selective_generator_page(request):
    return HttpResponse("MNIST")


@csrf_exempt
def rest_learn_curve(request):
    from MLGallery.feed_forward.polynomial import view_handler
    return view_handler.get_view(request)
