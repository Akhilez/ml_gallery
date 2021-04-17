from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from mlg.settings import logger
import numpy as np
from gym_nine_mens_morris.envs.nine_mens_morris_env import NineMensMorrisEnv, Pix


@api_view(["POST"])
def handle_step_request(request):
    board = request.data.get("board")
    mens = request.data.get("mens")
    me = request.data.get("me")
    action_position = request.data.get("actionPosition")
    move_position = request.data.get("movePosition")
    kill_position = request.data.get("killPosition")

    logger.info(f"Hey. Board: {board}")
    logger.info(f"Hey. mens: {mens}")
    logger.info(f"Hey. me: {me}")
    logger.info(f"Hey. action: {action_position}")
    logger.info(f"Hey. move: {move_position}")
    logger.info(f"Hey. kill at: {kill_position}")

    env = NineMensMorrisEnv()
    env.reset()
    env.board = np.array(board)
    env.mens = np.array(mens)
    env.player = Pix.B if me == "b" else Pix.W

    move = (
        env.get_move_from_position(action_position, move_position)
        if move_position
        else move_position
    )

    state, reward, is_done, info = env.step(action_position, move, kill_position)

    return Response(
        {"state": state, "reward": reward, "done": is_done, "info": info},
        status=status.HTTP_200_OK,
    )
