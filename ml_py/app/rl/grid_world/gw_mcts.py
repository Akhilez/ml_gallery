import numpy as np
import copy
import torch

from gym_grid_world.envs import GridWorldEnv

size = 4
mode = "static"
MAX_STEPS = 50

env = GridWorldEnv(size=size, mode=mode)


class MctsNode:
    def __init__(self, parent=None, action: int = None):
        self.parent = parent
        self.children = None
        self.action = action
        self.n = 0
        self.wins = 0
        self.is_terminal = False

        temp_env = GridWorldEnv(size, mode)
        temp_env.reset()

        self.state: np.ndarray = (
            copy.deepcopy(parent.state) if parent is not None else temp_env.state
        )

        if action is not None:
            temp_env.set_state(self.state)
            self.state, _, self.is_terminal, _ = temp_env.step(action)

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


def select(node: MctsNode) -> MctsNode:
    if node.n != 0 and node.children is not None:
        scores = np.array([uct(child) for child in node.children])
        return select(node.children[np.argmax(scores)])
    return node


def expand(node: MctsNode) -> MctsNode:
    if GridWorldEnv.is_done(node.state):
        return node
    if node.children is None:
        node.children = [MctsNode(node, action) for action in range(4)]
    return np.random.choice(node.children)


def rollout(state: np.ndarray, attempt: int = 0) -> float:
    if attempt >= MAX_STEPS:
        return 0
    if env.is_done(state):
        if env.has_won(state):
            return 1
        return 0
    env_ = GridWorldEnv(size, mode)
    env_.reset()
    env_.set_state(state)
    state, _, _, _ = env_.step(np.random.choice(range(4)))
    return rollout(state, attempt + 1)


def backprop(node: MctsNode, win_score: float) -> None:
    node.n += 1
    node.wins += win_score
    if node.parent is not None:
        backprop(node.parent, win_score)


def iter_mcts():
    node = select(tree)
    node = expand(node)
    win = rollout(copy.deepcopy(node.state))
    backprop(node, win)
    print(".", end="")


def uct(node):
    # sqrt(log(parent_visit) / child_visit)
    return np.sqrt(np.log(node.parent.n) / (node.n + 0.00001))


def mcts_player():
    [iter_mcts() for _ in range(100)]

    probs = torch.tensor(
        [0 if child.n == 0 else child.wins / child.n for child in tree.children]
    )
    print(probs)
    probs = torch.softmax(probs, 0)
    print(tree, probs)
    return tree.children[probs.argmax(0)].action


def play(player, render=False):
    global tree

    env.reset()
    tree = MctsNode()
    expand(tree)

    if render:
        env.render()

    step = 0
    while not env.done and step < MAX_STEPS:
        action = player()
        _, reward, _, _ = env.step(action)
        if render:
            env.render()

        tree = MctsNode()
        tree.state = env.state
        # tree = tree.children[action]

        step += 1

    print(reward)


if __name__ == "__main__":
    print(play(mcts_player, True))
