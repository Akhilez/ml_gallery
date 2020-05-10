from django.urls import path
from experiments import views


app_name = 'experiments'


urlpatterns = [

    # ---------------- Chat -------------------
    path('chat', views.chat, name='chat'),
    path('chat/<room>', views.chat_room, name='chat_room'),
    # -----------------------------------------

]
