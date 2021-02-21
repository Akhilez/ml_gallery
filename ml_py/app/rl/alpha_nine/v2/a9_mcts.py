from gym_nine_mens_morris.envs.nmm_v2 import NineMensMorrisEnvV2
import numpy as np
import copy
import torch

env = NineMensMorrisEnvV2()
learner = 1


class MctsNode:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.children = None
        self.action = action
        self.n = 0
        self.wins = 0
        self.turn = -1 if parent is None else -parent.turn
        self.state = (
            copy.deepcopy(parent.state) if parent is not None else env.get_empty_state()
        )
        self.illegal = False
        self.is_terminal = False
        if action is not None:
            if NineMensMorrisEnvV2.is_illegal(self.state, self.turn, action):
                self.illegal = True
            else:
                env_ = NineMensMorrisEnvV2()
                env_.turn = self.turn
                env_.state = self.state
                self.state, _, self.is_terminal, _ = env_.step(action)

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


def uct(node):
    return np.sqrt(np.log(node.parent.n) / (node.n + 0.00001))


def select(node):
    if node.n != 0 and node.children is not None:
        scores = np.array([uct(child) for child in node.children])
        return select(node.children[np.argmax(scores)])
    return node


def expand(node):
    if env.is_done(node.state):
        return node
    if node.children is None:
        flattened_actions = env.flatten_actions(
            env.get_legal_actions_(node.state, node.turn)
        )
        node.children = [MctsNode(node, action) for action in flattened_actions]
    return np.random.choice(node.children)


def rollout(state, turn):
    env = NineMensMorrisEnvV2()
    env.reset()
    env.state = state
    env.turn = turn
    winner = env.is_done(env.state)
    if winner:
        return max(0, winner * learner)
    actions = env.flatten_actions(env.get_legal_actions_(state, turn))
    action = actions[np.random.choice(range(len(actions)))]
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
    win = rollout(copy.deepcopy(node.state), -node.turn)
    backprop(node, win)
    print("`", end="")


def mcts_player(env):
    global tree
    # if all(tree.state != env.state):
    # tree = find_state_node(tree.state)
    # pass

    [iter_mcts() for _ in range(8)]

    probs = torch.tensor(
        [0 if child.n == 0 else child.wins / child.n for child in tree.children]
    )
    print(tree, probs)
    probs = torch.softmax(probs, 0)
    return tree.children[probs.argmax(0)].action


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


if __name__ == "__main__":
    print(play(mcts_player, random_player, -1, True))
    print("----")
    print(play(mcts_player, random_player, 1, True))
