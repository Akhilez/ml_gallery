from datetime import datetime
from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from gym_grid_world.envs import GridWorldEnv

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GWPgModel(nn.Module):
    def __init__(self, size: int, units: List[int]):
        super().__init__()
        self.size = size

        self.first = nn.Sequential(
            nn.Conv2d(4, units[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.hidden = nn.ModuleList([nn.Sequential(
            nn.Conv2d(units[i], units[i + 1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3)
        ) for i in range(len(units) - 1)])

        self.out = nn.Linear(self.size * self.size * units[-1], 4)

    def forward(self, x):
        x = self.first(x)
        for hidden in self.hidden:
            x = hidden(x)
        x = x.flatten(1)
        return self.out(x)

    @staticmethod
    def convert_inputs(envs):
        inputs = np.array([env.state for env in envs])
        return torch.tensor(inputs).double().to(device)


lrs = [1e-3]
depths = [2]
unitss = [50]

lr = lrs[0]
depth = depths[0]
units = unitss[0]

gamma_returns = 0.80
gamma_credits = 0.95
total_episodes = 500
n_env = 50
current_episode = 1
max_steps = 100
size = 4
mode = 'random'

model = GWPgModel(size, [units for _ in range(depth)]).double().to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)
writer = SummaryWriter(f'./runs/gw_policy_grad__{int(datetime.now().timestamp())}')
envs = [GridWorldEnv(size=size, mode=mode) for i in range(n_env)]
stats_e = []
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
    won = [None for _ in envs]


def sample_action(probs):
    # Softmax
    tau = max((1 / (np.log(current_episode) * 5 + 0.0001)), 0.7)
    probs = F.gumbel_softmax(probs, tau=tau, dim=0)
    # probs = F.softmax(probs, dim=0)

    # Add noise
    # noise = torch.rand(len(probs)) * 0.1
    # probs = probs + noise

    # Subsample legal probs
    # legal_probs = probs[legal_idx]

    # Sample action idx
    # if len(legal_probs) == 0:
    #     return 0, 0, 0
    action = torch.multinomial(probs, 1)[0]

    return action, probs[action]


def run_time_step(yh):
    for i in range(n_env):

        if envs[i].done:
            continue

        action, prob = sample_action(yh[i])
        _, reward, done, _ = envs[i].step(action)
        # envs[i].render()

        stats_e[i].append({'reward': reward, 'prob': prob})
        won[i] = done and envs[i].won


def learn():
    loss = torch.tensor(0).double().to(device)
    rewards_list = []
    for i in range(n_env):
        probs = [stat['prob'] for stat in stats_e[i]]
        if len(probs) == 0:
            continue
        probs = torch.stack(probs)
        rewards = [stat['reward'] for stat in stats_e[i]]
        returns = get_returns(rewards)
        credits = get_credits(len(rewards))

        loss += torch.sum(probs * credits * returns)
        rewards_list.append(np.mean(rewards))

    loss = -1 * loss / n_env

    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f"loss: {loss}")
    writer.add_scalar('Training loss', loss.item(), global_step=current_episode)
    writer.add_scalar('Mean Rewards', np.mean(rewards_list), global_step=current_episode)

    # losses.append(loss.item())


def run_episode():
    # Reset envs
    reset_episode()
    step = 0

    while not all([env.done for env in envs]) and step < max_steps:
        # Predict actions

        x = GWPgModel.convert_inputs(envs)
        yh = model(x)

        run_time_step(yh)
        step += 1

    if step == max_steps:
        for i in range(n_env):
            if not envs[i].done:
                stats_e[i].append({'reward': -10, 'prob': torch.tensor(0)})

    learn()


while current_episode <= total_episodes:
    run_episode()
    print('.', end='')
    current_episode += 1


def play():
    env = GridWorldEnv(4, mode)
    env.reset()
    env.render()

    while not env.done:
        x = GWPgModel.convert_inputs([env])
        yh = model(x)
        yh = F.softmax(yh, 1)
        action = yh[0].argmax(0)

        _, reward, done, _ = env.step(action)

        env.render()


play()
