# from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    #    path('admin/', admin.site.urls),
    path('exp/', include('experiments.urls', namespace='exp')),
    path('', include('app.urls', namespace='mlg')),
]
