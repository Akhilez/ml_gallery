import numpy as np
from gym_nine_mens_morris.envs.nmm_v2 import NineMensMorrisEnvV2


def play(p1, p2, render=False):
    env = NineMensMorrisEnvV2()
    env.reset()

    while not env.done:
        p = p1 if env.turn == 1 else p2
        action = p(env)
        print(f"Action: {action}")
        state, reward, done, info = env.step(action)
        print(reward, done, info)
        if render:
            env.render()

    return env.winner


def random_player(env):
    actions, opponents = env.get_legal_actions()
    action = actions[np.random.choice(range(len(actions)))]
    if action[1] is not None and len(action[1]) > 0:
        move_idx = np.random.choice(range(len(action[1])))
        action[1] = action[1][move_idx]
        action[2] = action[2][move_idx]
    if action[2]:
        action[2] = np.random.choice(opponents)
    return action


print(play(random_player, random_player, True))
