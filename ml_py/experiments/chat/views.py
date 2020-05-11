from django.shortcuts import render


def chat(request):
    return render(request, 'experiments/chat/index.html', {})


def chat_room(request, room):
    return render(request, 'experiments/chat/room.html', {'room_name': room})
