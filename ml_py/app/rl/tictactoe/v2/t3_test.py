from gym_tic_tac_toe.envs.t3_v2 import TicTacToeEnvV2
import numpy as np


def play(p1, p2):
    env = TicTacToeEnvV2()
    env.reset()

    while not env.done:
        p = p1 if env.turn > 0 else p2
        _, _, done, _ = env.step(p(env))
        env.render()

    print(f"Won = {env.winner}")


def random_player(env):
    actions = env.get_legal_actions()
    return np.random.choice(actions)


play(random_player, random_player)
