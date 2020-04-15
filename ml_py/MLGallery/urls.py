from django.conf.urls import url
from django.urls import path

from MLGallery import views

__author__ = 'Akhilez'

app_name = 'quiz'

urlpatterns = [
    path('selective_generator', views.selective_generator_page, name='selective_generator'),
    path('', views.home, name='home'),
]
