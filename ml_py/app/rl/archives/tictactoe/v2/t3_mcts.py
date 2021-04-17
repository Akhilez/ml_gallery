from gym_tic_tac_toe.envs import TicTacToeEnvV2
import numpy as np
import copy
import torch

env = TicTacToeEnvV2()
learner = 1


class MctsNode:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.children = None
        self.action = action
        self.n = 0
        self.wins = 0
        self.turn = -1 if parent is None else -parent.turn
        self.state = copy.deepcopy(parent.state) if parent is not None else np.zeros(9)
        self.illegal = False
        if action is not None:
            if self.state[action] != 0:
                self.illegal = True
            else:
                self.state[action] = self.turn

    def __str__(self):
        return f"({self.wins}/{self.n})"

    def __repr__(self):
        return str(self)


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
    if node.n != 0 and node.children is not None:
        legal_actions = env.get_legal_actions(node.state)
        scores = np.array([uct(child) for child in node.children])
        scores = scores[legal_actions]
        idx = legal_actions[np.argmax(scores)]
        return select(node.children[idx])
    return node


def expand(node):
    if env.is_done(node.state):
        return node
    if node.children is None:
        node.children = [MctsNode(node, action) for action in range(9)]
    idx = np.random.choice(env.get_legal_actions())
    return node.children[idx]


def rollout(state, turn):
    env = TicTacToeEnvV2()
    env.reset()
    env.state = state
    env.turn = turn
    if env.is_done():
        is_winner = env.is_winner()
        if is_winner is not None:
            winner = turn if is_winner else -turn
            return max(0, winner * learner)
        return 0
    actions = env.get_legal_actions()
    action = np.random.choice(actions)
    state, _, done, _ = env.step(action)
    return rollout(state, env.turn)


def backprop(node, win):
    node.n += 1
    node.wins += win
    if node.parent is not None:
        backprop(node.parent, win)


def iter_mcts():
    node = select(tree)
    node = expand(node)
    win = rollout(copy.deepcopy(node.state), node.turn)
    backprop(node, win)


def print_tree(nodes, max_depth=4):
    if max_depth < 1:
        return
    new_nodes = []
    for siblings in nodes:
        if siblings is None:
            continue
        for node in siblings:
            if node.parent is None:
                print(node)
            if node.children is not None:
                children = [child for child in node.children if child is not None]
                new_nodes.append(children)
            else:
                new_nodes.append(None)
    if len(new_nodes) != 0:
        print(new_nodes)
        return print_tree(new_nodes, max_depth - 1)


def find_state_node(state):
    return tree


def mcts_player(env):
    global tree
    if all(tree.state != env.state):
        tree = find_state_node(tree.state)

    [iter_mcts() for _ in range(800)]

    probs = torch.tensor(
        [0 if child.n == 0 else child.wins / child.n for child in tree.children]
    )
    print(tree, probs)
    legal_actions = env.get_legal_actions()
    probs = probs[legal_actions]
    probs = torch.softmax(probs, 0)
    idx = probs.argmax(0)
    action = legal_actions[idx]
    return action


def random_player(env):
    actions = env.get_legal_actions()
    return np.random.choice(actions)


def play(mcts_p, other_p, mcts_turn=1, render=False):
    global tree
    global learner

    env.reset()
    learner = mcts_turn
    tree = MctsNode()
    if env.turn != mcts_turn:
        expand(tree)

    while not env.done:
        p = mcts_p if env.turn == mcts_turn else other_p
        action = p(env)
        env.step(action)
        if render:
            env.render()

        # Continue with the same tree
        # tree = tree.children[action] if tree.children is not None else tree
        # tree.parent = None

        tree = MctsNode()
        tree.state = env.state
        tree.turn = -env.turn

    return env.winner


print(play(mcts_player, random_player, -1, True))
print("----")
print(play(mcts_player, random_player, 1, True))
