from django.urls import re_path

from MLGallery.regressors.polynomial.consumer import PolyRegConsumer

websocket_urlpatterns = [
    re_path(r'ws/poly_reg', PolyRegConsumer),
]
