from django.urls import re_path

from experiments.chat.consumers import ChatConsumer

websocket_urlpatterns = [
    re_path(r'ws/chat/(?P<room_name>\w+)$', ChatConsumer),
]
