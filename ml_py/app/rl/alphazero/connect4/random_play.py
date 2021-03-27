from pettingzoo.classic.connect_four import connect_four
import numpy as np


def random_player(env: connect_four.raw_env) -> int:
    state, reward, is_done, info = env.last()
    legal_actions = state["action_mask"]
    action = np.random.choice(legal_actions.nonzero()[0])
    return action


def play(player1, player2, render=False):
    env = connect_four.env()
    env.reset()
    if render:
        env.render()

    while not env.dones[env.agent_selection]:
        player = player1 if env.agent_selection == env.possible_agents[0] else player2

        action = player(env)

        env.step(action)

        if render:
            env.render()

    print(env.rewards)


play(random_player, random_player, True)
