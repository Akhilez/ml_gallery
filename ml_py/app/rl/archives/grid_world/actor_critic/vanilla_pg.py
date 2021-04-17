from datetime import datetime
from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from gym_grid_world.envs import GridWorldEnv
from lib.nn_utils import save_model
from settings import BASE_DIR, device

CWD = f"{BASE_DIR}/app/rl/grid_world/actor_critic"


class GwAcModel(nn.Module):
    def __init__(self, size: int, units: List[int]):
        super().__init__()
        self.size = size

        self.first = nn.Sequential(
            nn.Conv2d(4, units[0], kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.3)
        )

        self.hidden = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(units[i], units[i + 1], kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )
                for i in range(len(units) - 1)
            ]
        )

        self.out = nn.Linear(self.size * self.size * units[-1], 4)

    def forward(self, x):
        x = self.first(x)
        for hidden in self.hidden:
            x = hidden(x)
        x = x.flatten(1)
        return self.out(x)

    @staticmethod
    def convert_inputs(envs):
        """
        Outputs a tensor of shape(batch, 4,4,4)
        """
        inputs = np.array([env.state for env in envs])
        return torch.tensor(inputs).double().to(device)


# ------------------ CONSTANTS -----------------------

lr = 0.001
depth = 2
units = 25

gamma_returns = 0.80
gamma_credits = 0.95

total_episodes = 1000
n_env = 50
max_steps = 100

grid_size = 4
env_mode = "random"

# -----------------------------------------------------

model = GwAcModel(grid_size, [units for _ in range(depth)]).double().to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)

envs = [GridWorldEnv(size=grid_size, mode=env_mode) for _ in range(n_env)]

current_episode = 1
stats_e = None
won = None

writer = SummaryWriter(
    f"{CWD}/runs/gw_policy_grad_LR{str(lr)[:7]}_{depth}x{units}_{int(datetime.now().timestamp())}"
)

envs[0].reset()
writer.add_graph(model, GwAcModel.convert_inputs(envs[:1]))

# -----------------------------------------------------


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


def get_credits(t: int):
    return torch.pow(gamma_credits, torch.arange(t).float()).flip(0).double().to(device)


def get_returns(rewards):
    total_t = len(rewards)
    returns = []
    prev_return = 0
    for t in range(total_t):
        prev_return = rewards[total_t - t - 1] + (gamma_returns * prev_return)
        returns.append(prev_return)
    return torch.tensor(list(reversed(returns))).double().to(device)


def get_final_reward():
    reward = 0
    n = 0
    for stat in stats_e:
        reward += stat[-1]["reward"]
        n += 1
    return reward / (n + 0.00001)


def main():
    global current_episode, stats_e, won
    while current_episode <= total_episodes:

        # Reset envs
        [env.reset() for env in envs]
        stats_e = [[] for _ in envs]
        won = [None for _ in envs]

        step = 0

        # ---------------- Monte carlo loop --------------------
        while not all([env.done for env in envs]) and step < max_steps:
            # Predict actions

            x = GwAcModel.convert_inputs(envs)
            yh = model(x)

            for i in range(n_env):

                if envs[i].done:
                    continue

                action, prob = sample_action(yh[i])
                _, reward, done, _ = envs[i].step(action)
                # envs[i].render()

                stats_e[i].append({"reward": reward, "prob": prob})
                won[i] = done and envs[i].won

            step += 1

        # -------- Negative reward when max steps reached --------
        if step == max_steps:
            for i in range(n_env):
                if not envs[i].done:
                    stats_e[i].append({"reward": -10, "prob": torch.tensor(0)})

        # --------- Finding loss for each env -----------
        loss = torch.tensor(0).double().to(device)
        rewards_list = []
        for i in range(n_env):
            probs = [stat["prob"] for stat in stats_e[i]]
            if len(probs) == 0:
                continue
            probs = torch.log(torch.stack(probs))
            rewards = [stat["reward"] for stat in stats_e[i]]
            returns = get_returns(rewards)
            credits = get_credits(len(rewards))

            loss += torch.sum(probs * credits * returns)
            rewards_list.append(np.mean(rewards))

        loss = -1 * loss / n_env

        # ----------- Optimization -----------

        optim.zero_grad()
        loss.backward()
        optim.step()

        # print(f"loss: {loss}")
        writer.add_scalar("Training loss", loss.item(), global_step=current_episode)
        writer.add_scalar(
            "Mean Rewards", np.mean(rewards_list), global_step=current_episode
        )

        # losses.append(loss.item())

        print(".", end="")
        current_episode += 1

    final_reward = get_final_reward()
    hparams = {"lr": lr, "depth": depth, "units": units}
    writer.add_hparams(hparams, {"final_reward": final_reward})
    writer.close()

    # play(model, cfg)

    save_model(model, CWD, "grid_world_pg")


if __name__ == "__main__":
    main()
