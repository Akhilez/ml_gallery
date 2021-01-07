from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from mlg.settings import logger


@api_view(['POST'])
def handle_step_request(request):
    board = request.data.get('board')
    mens = request.data.get('mens')
    me = request.data.get('me')
    action_position = request.data.get('actionPosition')
    move = request.data.get('move')
    kill_position = request.data.get('killPosition')

    logger.info(f'Hey. Board: {board}')
    logger.info(f'Hey. mens: {mens}')
    logger.info(f'Hey. me: {me}')
    logger.info(f'Hey. action: {action_position}')
    logger.info(f'Hey. move: {move}')
    logger.info(f'Hey. kill at: {kill_position}')

    return Response({'state': [board, mens], 'reward': 0, 'done': False, 'info': None}, status=status.HTTP_200_OK)
