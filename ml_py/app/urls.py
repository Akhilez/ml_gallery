from django.conf.urls import url
from django.urls import path

from app import views

__author__ = 'Akhilez'

app_name = 'quiz'

urlpatterns = [
    path('selective_generator', views.selective_generator_page, name='selective_generator'),
    path('ajax/learn_curve', views.rest_learn_curve, name='rest-learn-curve'),
    path('', views.home, name='home'),
]
