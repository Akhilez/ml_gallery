from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from mlg.settings import logger


@api_view(['POST'])
def handle_step_request(request):
    board = request.data.get('board')
    mens = request.data.get('mens')
    me = request.data.get('me')

    logger.info(f'Hey. Board: {board}')
    logger.info(f'Hey. mens: {mens}')
    logger.info(f'Hey. me: {me}')

    return Response({}, status=status.HTTP_200_OK)
