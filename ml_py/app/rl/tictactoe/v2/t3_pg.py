import matplotlib.pyplot as plt
import copy
from typing import List
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from gym_tic_tac_toe.envs import TicTacToeEnvV2

device = "cuda" if torch.cuda.is_available() else "cpu"


class T3LinearModel(nn.Module):
    def __init__(self, units: List[int]):
        super().__init__()

        self.first = nn.Sequential(
            nn.Linear(9 * 3, units[0]), nn.ReLU(), nn.Dropout(0.3)
        )

        self.hidden = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(units[i], units[i + 1]), nn.ReLU(), nn.Dropout(0.3)
                )
                for i in range(len(units) - 1)
            ]
        )

        self.out = nn.Linear(units[-1], 9)

    def forward(self, x):
        x = self.first(x)
        for hidden in self.hidden:
            x = hidden(x)
        return self.out(x)

    @staticmethod
    def convert_inputs(x):
        # x: shape(n, 9)
        # output: shape(n, 27)
        inputs = []
        for xb in x:
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


model = T3LinearModel([50, 50]).double().to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

gamma_returns = 0.5
gamma_credits = 0.5


total_episodes = 1000
n_env = 100
buffer_reset_size = 100
current_episode = 1

envs = [TicTacToeEnvV2() for i in range(n_env)]
prev_models = [copy.deepcopy(model)]
prev_model = prev_models[0]
prev_model.eval()
stats_e = []
learners = []
losses = []
won = []


def reset_episode():
    global stats_e
    global learners
    global won

    [env.reset() for env in envs]
    stats_e = [[] for _ in envs]
    learners = randomize_learners()
    won = [None for _ in envs]


def randomize_learners():
    turns = torch.rand(n_env)
    turns[turns > 0.5] = 1
    turns[turns <= 0.5] = -1
    if all(turns > 0) or all(turns < 0):
        return randomize_learners()
    return turns


def sample_action(yh, i):
    # Get legal actions
    # Apply some softmax
    # sample an action

    # Sample legal
    legal_actions = envs[i].get_legal_actions()
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


def log_stats():
    wins = sum([1 for i in range(n_env) if won[i]])
    draws = sum([1 for i in range(n_env) if won[i] is None])
    loses = len(won) - wins - draws

    loss = np.mean(losses)

    print(f"Ep: {current_episode}\tLoss: {loss}\tW: {wins}\tL: {loses}\tD: {draws}")


def reset_buffer():
    global losses
    global prev_model
    global prev_models

    losses = []
    prev_models = prev_models[-10:]
    prev_models.append(copy.deepcopy(model))
    prev_model = prev_models[np.random.choice(len(prev_models), 1)[0]]
    prev_model.eval()


def learn():
    loss = torch.tensor(0).double().to(device)
    for i in range(n_env):
        rewards = [stat["reward"] for stat in stats_e[i]]
        returns = get_returns(rewards)
        probs = torch.stack([stat["prob"] for stat in stats_e[i]])
        credits = get_credits(len(rewards))

        loss += torch.sum(probs * credits * returns)

    loss = -1 * loss / n_env

    optim.zero_grad()
    loss.backward()
    optim.step()

    losses.append(loss.item())


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
            stats_e[i].append({"reward": reward, "prob": prob})

        if done and envs[i].winner is not None:
            won[i] = envs[i].winner == learners[i]


def run_episode():
    # Reset envs
    reset_episode()

    while not all([env.done for env in envs]):
        # Predict actions

        x = T3LinearModel.convert_inputs([env.state for env in envs])
        yh = model(x)
        with torch.no_grad():
            yo = prev_model(x)

        run_time_step(yh, yo)

    learn()


while current_episode <= total_episodes:
    run_episode()
    print(".", end="")
    if current_episode % buffer_reset_size == 0:
        log_stats()
        reset_buffer()
    current_episode += 1


def AIPlayer(env):
    x = T3LinearModel.convert_inputs([env.state])
    with torch.no_grad():
        yh = model(x)
    legal_actions = env.get_legal_actions()
    legal_yh = yh[0][legal_actions]
    idx = int(legal_yh.argmax(0))
    return legal_actions[idx]


def play(p1, p2, render=False):
    env = TicTacToeEnvV2()
    env.reset()

    while not env.done:
        p = p1 if env.turn > 0 else p2
        _, _, done, _ = env.step(p(env))
        if render:
            env.render()

    return env.winner


def random_player(env):
    actions = env.get_legal_actions()
    return np.random.choice(actions)


def play_out(p1, p2, plays):
    results = [play(p1, p2) for _ in range(plays)]
    p1 = sum([1 for r in results if r == 1])
    p2 = sum([1 for r in results if r == -1])
    d = sum([1 for r in results if r is None])

    return p1 / plays, p2 / plays, d / plays


print(play_out(random_player, AIPlayer, 100))
print(play_out(AIPlayer, random_player, 100))

print(play(random_player, AIPlayer, render=True))
print(play(AIPlayer, random_player, render=True))
