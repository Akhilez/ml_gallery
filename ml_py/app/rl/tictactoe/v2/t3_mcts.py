from gym_tic_tac_toe.envs import TicTacToeEnvV2
import numpy as np
import torch
import copy

env = TicTacToeEnvV2()

learner = 1


class MctsNode:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.children = None
        self.action = action
        self.n = 0
        self.wins = 0
        self.is_learner = True if parent is None else not parent.is_learner
        self.state = copy.deepcopy(parent.state) if parent is not None else np.zeros(9)
        if action is not None:
            self.state[action] = learner if self.is_learner else -learner


"""
- For each iteration, you must select, expand, rollout and backprop
- Select:
  - If not visited, expand
  - Find UCT1 for all children
  - Select top UCT1 child, go to step 1.1
- Expand:
  - create all children nodes
  - select random action
- Rollout:
  - play random actions till win/lose/draw
- Backprop:
  - If won, update all nodes above with 1/1 else 0/1
"""

tree = MctsNode()


def uct(node):
    return np.sqrt(np.log(node.parent.n) / (node.n + 0.00001))


def select(node):
    if node.n != 0 and node.parent is not None:
        idx = np.argmax([uct(child) for child in node.children])
        return select(node.children[idx])
    return node


def expand(node):
    node.children = [MctsNode(node, action) for action in range(9)]
    idx = np.random.choice(range(9))
    return node.children[idx]


def rollout(state, turn):
    env = TicTacToeEnvV2()
    env.reset()
    env.state = state
    env.turn = turn
    actions = env.get_legal_actions()
    action = np.random.choice(actions)
    state, _, done, _ = env.step(action)
    if done and env.winner is not None:
        return env.winner * learner
    return rollout(state, env.turn)


def backprop(node, win):
    node.n += 1
    node.wins += win
    if node.parent is not None:
        backprop(node.parent, win)


def iter_mcts():
    node = select(tree)
    node = expand(node)
    win = rollout(node.state, learner if node.is_learner else -learner)
    backprop(node, win)


def print_tree(root):
    pass


iterations = 10
for i in range(iterations):
    iter_mcts()

print_tree(tree)
