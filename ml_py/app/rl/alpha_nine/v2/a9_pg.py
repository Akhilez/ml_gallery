from typing import List
import torch
from torch import nn
import copy
import numpy as np
from torch.nn import functional as F
from gym_nine_mens_morris.envs.nmm_v2 import NineMensMorrisEnvV2
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"


class A9PgModel(nn.Module):
    """
    1. 24 * 3 = 72
    2. ...
    3. 24: pos1, 24: pos2, 24 left, 24 up, 24 right, 24 left, 24 kill = 7 * 24 = 168
    """

    def __init__(self, units: List[int]):
        super().__init__()

        self.first = nn.Sequential(
            nn.Linear(24 * 3, units[0]), nn.ReLU(), nn.Dropout(0.3)
        )

        self.hidden = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(units[i], units[i + 1]), nn.ReLU(), nn.Dropout(0.3)
                )
                for i in range(len(units) - 1)
            ]
        )

        self.out = nn.Linear(units[-1], 7 * 24)

    def forward(self, x):
        x = self.first(x)
        for hidden in self.hidden:
            x = hidden(x)
        return self.out(x).view((-1, 7, 24))

    @staticmethod
    def convert_inputs(states):
        # state: shape(n, 24), mens
        # output: shape(n, 72)
        inputs = []
        for state in states:
            xb = state[0]
            inputs_b = []
            for xi in xb:
                if xi == 1:
                    inputs_b.extend([1, 0, 0])
                elif xi == -1:
                    inputs_b.extend([0, 0, 1])
                else:
                    inputs_b.extend([0, 1, 0])
            inputs.append(inputs_b)
        return torch.tensor(inputs).double().to(device)

    @staticmethod
    def convert_outputs(op):
        # op: shape(n, 7, 24)
        # output: tuple((n, 24), (n, 24), (n, 4, 24), (n, 24))
        pos1 = op[:, 0]
        pos2 = op[:, 1]
        moves = op[:, 2:6]
        kill = op[:, -1]

        return pos1, pos2, moves, kill


lrs = [1e-3]
depths = [1]
unitss = [2]

lr = lrs[0]
depth = depths[0]
units = unitss[0]


gamma_returns = 0.99
gamma_credits = 0.99
total_episodes = 10
n_env = 2
buffer_reset_size = 1
current_episode = 1

model = A9PgModel([units for _ in range(depth)]).double().to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)
writer = SummaryWriter(f"./runs/9mm_policy_grad__{int(datetime.now().timestamp())}")
envs = [NineMensMorrisEnvV2() for i in range(n_env)]
prev_models = [copy.deepcopy(model)]
prev_model = prev_models[0]
prev_model.eval()
stats_e = []
learners = []
won = []

"""
Log the following:
For each episode:
  - Episode length (min, max, avg)
  - Win / lose
  - sum of rewards
  - loss
For every nth episodes, for each timestep:
  - Policy histogram
  - value
"""


def randomize_learners():
    turns = torch.rand(n_env)
    turns[turns > 0.5] = 1
    turns[turns <= 0.5] = -1
    if all(turns > 0) or all(turns < 0):
        return randomize_learners()
    return turns


def get_credits(t):
    credits = []
    prev_credit = 1
    for i in range(t):
        credits.append(prev_credit)
        prev_credit *= gamma_credits
    return torch.tensor(list(reversed(credits))).double().to(device)


def get_returns(rewards):
    total_t = len(rewards)
    returns = []
    prev_return = 0
    for t in range(total_t):
        prev_return = rewards[total_t - t - 1] + (gamma_returns * prev_return)
        returns.append(prev_return)
    return torch.tensor(list(reversed(returns))).double().to(device)


def reset_episode():
    global stats_e
    global learners
    global won

    [env.reset() for env in envs]
    stats_e = [[] for _ in envs]
    learners = randomize_learners()
    won = [None for _ in envs]


def reset_buffer():
    # global losses
    global prev_model
    global prev_models

    # losses = []
    prev_models = prev_models[-10:]
    prev_models.append(copy.deepcopy(model))
    prev_model = prev_models[np.random.choice(len(prev_models), 1)[0]]
    prev_model.eval()


def sample_from_probs(probs, legal_idx):

    # Add noise
    # noise = torch.rand(len(probs)) * 0.01
    # probs = probs + noise

    # Softmax
    tau = max((1 / (np.log(current_episode)) * 5 + 0.0001), 0.7)
    probs = F.gumbel_softmax(probs, tau=tau, dim=0)

    # Subsample legal probs
    legal_probs = probs[legal_idx]

    # Sample action idx
    if len(legal_probs) == 0:
        return 0, 0, 0
    action_idx = torch.multinomial(legal_probs, 1)[0]

    action_prob = legal_probs[action_idx]
    legal_action_idx = legal_idx[action_idx]
    return legal_action_idx, action_prob, action_idx


def sample_action(yh, env):
    """
    yh: A tensor of shape (7, 24)
    i: current batch_number
    return: A tuple (
        action: (pos, move, kill),
        probability distribution indices: (pos_idx, move_idx, kill_idx)
    )
    Steps:
        - Determine legal actions and subsample them.
          - Get all legal POS
          - For each legal POS, get legal moves.
          - Get all legal kills
        - Sample legal pos from probs
        - Sample legal moves from the selected pos
        - Sample legal kill
    Legal action format given:
        [(pos, moves, bools)], [kill positions]
    """
    is_phase_1 = env.is_phase_1()
    legal_actions = env.get_legal_actions()
    if len(legal_actions[0]) == 0:
        return (0, None, None), torch.tensor(0)

    pos, move, kill, (pos_prob, move_prob, kill_prob) = (
        None,
        None,
        None,
        torch.tensor([0, 0, 0]),
    )

    if is_phase_1:
        pos, pos_prob, pos_idx = sample_from_probs(
            yh[0], [action[0] for action in legal_actions[0]]
        )

        if legal_actions[0][pos_idx][2]:
            kill, kill_prob, kill_idx = sample_from_probs(yh[-1], legal_actions[1])

        return (pos, None, kill), sum((pos_prob, move_prob, kill_prob)) / 3

    pos, pos_prob, pos_idx = sample_from_probs(
        yh[1], [action[0] for action in legal_actions[0]]
    )

    move, move_prob, move_idx = sample_from_probs(
        yh[2:6].T[pos_idx], legal_actions[0][pos_idx][1]
    )

    if legal_actions[0][pos_idx][2][move_idx]:
        kill, kill_prob, kill_idx = sample_from_probs(yh[-1], legal_actions[1])

    return (pos, move, kill), sum((pos_prob, move_prob, kill_prob)) / 3


def run_time_step(yh, yo):
    for i in range(n_env):

        if envs[i].done:
            continue

        is_learners_turn = learners[i] == envs[i].turn
        yi = yh[i] if is_learners_turn else yo[i]
        action, prob = sample_action(yi, envs[i])
        _, reward, done, _ = envs[i].step(action)
        # envs[i].render()

        if is_learners_turn:
            stats_e[i].append({"reward": reward, "prob": prob})

        if done and envs[i].winner is not None:
            won[i] = envs[i].winner == learners[i]


def learn():
    loss = torch.tensor(0).double().to(device)
    for i in range(n_env):
        probs = [stat["prob"] for stat in stats_e[i]]
        if len(probs) == 0:
            continue
        probs = torch.stack(probs)
        rewards = [stat["reward"] for stat in stats_e[i]]
        returns = get_returns(rewards)
        credits = get_credits(len(rewards))

        loss += torch.sum(probs * credits * returns)

    loss = -1 * loss / n_env

    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f"loss: {loss}")
    writer.add_scalar("Training loss", loss.item(), global_step=current_episode)

    # losses.append(loss.item())


def run_episode():
    # Reset envs
    reset_episode()

    while not all([env.done for env in envs]):
        # Predict actions

        x = A9PgModel.convert_inputs([env.state for env in envs])
        yh = model(x)
        with torch.no_grad():
            yo = prev_model(x)

        run_time_step(yh, yo)

    learn()
    reset_buffer()


while current_episode <= total_episodes:
    run_episode()
    print(".", end="")
    current_episode += 1
