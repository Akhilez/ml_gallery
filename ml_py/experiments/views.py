

# --------------------- Chat -----------------------


def chat(request):
    from .chat.views import chat as chat_view
    return chat_view(request)


def chat_room(request, room):
    from .chat.views import chat_room as chat_room_view
    return chat_room_view(request, room)

# --------------------------------------------------
