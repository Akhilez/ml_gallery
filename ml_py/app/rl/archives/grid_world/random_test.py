from gym_grid_world.envs.grid_world_env import GridWorldEnv
import numpy as np


def play(player):
    env = GridWorldEnv(size=4, mode="random")
    env.reset()
    max_episode_len = 100

    step = 0
    while not env.done and step < max_episode_len:
        _, _, done, _ = env.step(player(env))
        env.render()
        step += 1

    print(f"Won = {env.won}")


def random_player(env):
    actions = env.get_legal_actions()
    return np.random.choice(actions)


play(random_player)
