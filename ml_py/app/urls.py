from django.urls import path

from app import views

__author__ = 'Akhilez'

app_name = 'quiz'

urlpatterns = [
    path('selective_generator', views.selective_generator_page, name='selective_generator'),
    path('ajax/learn_curve', views.rest_learn_curve, name='rest-learn-curve'),
    path('next_char', views.next_char, name='next_char'),
    path('positional_cnn', views.positional_cnn, name='positional_cnn'),
    path('', views.home, name='home'),
]
