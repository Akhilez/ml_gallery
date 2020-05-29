from django.urls import re_path

from MLGallery.feed_forward.polynomial.websocket_consumer import PolyRegConsumer

websocket_urlpatterns = [
    re_path(r'ws/learn_curve', PolyRegConsumer),
]
