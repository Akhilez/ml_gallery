from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from experiments.chat import routing as chat_routing
from MLGallery.regressors.polynomial import routing as poly_reg_routing


application = ProtocolTypeRouter({
    # (http->django views is added by default)
    'websocket': AuthMiddlewareStack(
        URLRouter(
            chat_routing.websocket_urlpatterns +
            poly_reg_routing.websocket_urlpatterns
        )
    ),
})
