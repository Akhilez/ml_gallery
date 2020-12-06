# from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    #    path('admin/', admin.site.urls),
    path('exp/', include('experiments.urls', namespace='exp')),
    path('api-auth/', include('rest_framework.urls')),
    path('', include('app.urls', namespace='mlg')),
]
