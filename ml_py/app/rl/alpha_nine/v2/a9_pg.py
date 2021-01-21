from typing import List
import torch
from torch import nn
import copy
import numpy as np
from torch.nn import functional as F
from gym_nine_mens_morris.envs.nmm_v2 import NineMensMorrisEnvV2

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class A9PgModel(nn.Module):
    """
    1. 24 * 3 = 72
    2. ...
    3. 24: pos1, 24: pos2, 24 left, 24 up, 24 right, 24 left, 24 kill = 7 * 24 = 168
    """
    def __init__(self, units: List[int]):
        super().__init__()

        self.first = nn.Sequential(
            nn.Linear(24 * 3, units[0]),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.hidden = nn.ModuleList([nn.Sequential(
            nn.Linear(units[i], units[i + 1]),
            nn.ReLU(),
            nn.Dropout(0.3)
        ) for i in range(len(units) - 1)])

        self.out = nn.Linear(units[-1], 7 * 24)

    def forward(self, x):
        x = self.first(x)
        for hidden in self.hidden:
            x = hidden(x)
        return self.out(x).view((7, 24))

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
        moves = op[:, 2: 6]
        kill = op[:, -1]

        return pos1, pos2, moves, kill


lrs = [1e-3]
depths = [1]
unitss = [2]

lr = lrs[0]
depth = depths[0]
units = unitss[0]


epochs = 1
gamma_returns = 0.99
gamma_credits = 0.99
total_episodes = 1
n_env = 2
buffer_reset_size = 1
current_episode = 1

model = A9PgModel([units for _ in range(depth)]).double().to(device)
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
  - Episode length
  - Win / lose
  - sum of rewards
  - loss
For every nth episodes, for each timestep:
  - Policy histogram
  - value
"""

optim = torch.optim.Adam(model.parameters(), lr=lr)


def randomize_learners():
    turns = torch.rand(n_env)
    turns[turns > 0.5] = 1
    turns[turns <= 0.5] = -1
    if all(turns > 0) or all(turns < 0):
        return randomize_learners()
    return turns


def reset_episode():
    global stats_e
    global learners
    global won

    [env.reset() for env in envs]
    stats_e = [[] for _ in envs]
    learners = randomize_learners()
    won = [None for _ in envs]


def convert_actions_to_indices(actions):
    pass


def sample_action(yh, i):
    # TODO: Implement
    # Get legal actions
    # Convert legal actions to indices
    # Determine if phase 1 or 2
    # Subsample probabilities
    # Apply some softmax
    # sample an action

    legal_actions = envs[i].get_legal_actions()
    legal_actions = convert_actions_to_indices(legal_actions)

    legal_yh = yh[legal_actions]

    # Add noise
    noise = torch.rand(len(legal_actions)) * 0.01
    legal_yh = legal_yh + noise

    # Softmax
    tau = max((1 / np.log(current_episode)) * 5, 0.7)
    legal_yh = F.gumbel_softmax(legal_yh, tau=tau, dim=0)

    # Sample
    sampled = torch.multinomial(legal_yh, 1)[0]

    return legal_actions[sampled], legal_yh[sampled]


def run_time_step(yh, yo):
    for i in range(n_env):

        if envs[i].done:
            continue

        is_learners_turn = learners[i] == envs[i].turn
        yi = yh[i] if is_learners_turn else yo[i]
        if not is_learners_turn:
            yi = torch.ones(9).double().to(device)
        action, prob = sample_action(yi, i)
        _, reward, done, _ = envs[i].step(action)

        if is_learners_turn:
            stats_e[i].append({'reward': reward, 'prob': prob})

        if done and envs[i].winner is not None:
            won[i] = envs[i].winner == learners[i]



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
    log_stats()
    reset_buffer()


while current_episode <= total_episodes:
    run_episode()
    print('.', end='')
    current_episode += 1

