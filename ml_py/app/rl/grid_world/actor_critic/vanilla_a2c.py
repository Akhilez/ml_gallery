from datetime import datetime
from typing import List, Dict, Union

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

        self.policy = nn.Linear(self.size * self.size * units[-1], 4)
        self.value = nn.Sequential(
            nn.Linear(self.size * self.size * units[-1], 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )

    def forward(self, x):
        x = self.first(x)
        for hidden in self.hidden:
            x = hidden(x)
        x = x.flatten(1)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value

    @staticmethod
    def convert_inputs(envs):
        """
        Outputs a tensor of shape(batch, 4,4,4)
        """
        inputs = np.array([env.state for env in envs])
        return torch.tensor(inputs).double().to(device)


def main():

    # ----------------- Hyper params -------------------

    # Env params
    GRID_SIZE = 4
    ENV_MODE = "random"

    # TRAINING_PARAMS
    EPOCHS = 10000
    BATCH_SIZE = 50
    MAX_MONTE_CARLO_STEPS = 50
    N_TRAIN_STEP = 4
    ARCHITECTURE = [50, 50]
    GAMMA_RETURNS = 0.75
    GAMMA_CREDITS = 0.75
    LEARNING_RATE = 1e-3

    # -------------- Setup other variables ----------------

    model = GwAcModel(GRID_SIZE, ARCHITECTURE).double().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    envs = [GridWorldEnv(size=GRID_SIZE, mode=ENV_MODE) for _ in range(BATCH_SIZE)]
    global_step = 0
    timestamp = int(datetime.now().timestamp())
    writer = SummaryWriter(f"{CWD}/runs/gw_ac_LR{str(LEARNING_RATE)[:7]}_{timestamp}")

    # Add model graph
    envs[0].reset()
    writer.add_graph(model, GwAcModel.convert_inputs(envs[:1]))

    # -------------- Training loop ----------------
    for epoch in range(EPOCHS):

        [env.reset() for env in envs]
        stats: List[List[Dict[str, Union[torch.Tensor, float]]]] = [[] for _ in envs]
        step = 0

        episode_rewards = []

        # -------------- Monte Carlo Loop ---------------------
        while True:

            # ----------- Predict policy and value -------------
            states = GwAcModel.convert_inputs(envs)
            policy, value = model(states)  # Shapes: ph: (batch, 4); vh: (batch, 1)

            # ------------ Sample actions -----------------
            tau = max((1 / (np.log(epoch) + 0.0001) * 5), 0.7)
            writer.add_scalar("tau", tau, global_step=global_step)
            policy = F.gumbel_softmax(policy, tau=tau, dim=1)
            actions = torch.multinomial(policy, 1).squeeze()  # shape: (batch)

            # ------------- Rewards from step ----------------
            for i in range(BATCH_SIZE):
                if not envs[i].done:
                    _, reward, _, _ = envs[i].step(actions[i])
                    stats[i].append(
                        {
                            "reward": reward,
                            "value": value[i][0],
                            "policy": policy[i][actions[i]],
                        }
                    )
                    episode_rewards.append(reward)

            # -------------- Termination conditions ------------------

            all_done = all([env.done for env in envs])
            has_timed_out = step >= MAX_MONTE_CARLO_STEPS
            n_step_ended = step % N_TRAIN_STEP == 0

            if has_timed_out:
                # Set unfinished env's reward to -10
                for i in range(BATCH_SIZE):
                    if not envs[i].done:
                        stats[i][-1]["reward"] = -10
                all_done = True

            if all_done or n_step_ended:
                # ----------- Add last state's value -------------
                states = GwAcModel.convert_inputs(envs)

                model.eval()
                with torch.no_grad():
                    _, value = model(states)
                model.train()

                for i in range(BATCH_SIZE):
                    stats[i].append({"value": value[i][0]})

                # -------------- LEARN -----------------
                # loss = naive_ac_loss(stats, GAMMA_RETURNS, GAMMA_CREDITS)
                loss = advantage_ac_loss(stats, GAMMA_RETURNS)

                optim.zero_grad()
                loss.backward()
                optim.step()

                # ------------ Logging ----------------
                writer.add_scalar("Training loss", loss.item(), global_step=global_step)
                global_step += 1

                # Clean up
                stats = [[] for _ in envs]

            if all_done:
                break
            writer.add_scalar(
                "Mean Rewards", np.mean(episode_rewards), global_step=global_step
            )
            step += 1

        print(".", end="")

    save_model(model, CWD, "grid_world_ac")


def naive_ac_loss(
    stats: List[List[Dict[str, Union[torch.Tensor, float]]]],
    gamma_returns: float,
    gamma_credits: float,
) -> torch.Tensor:

    loss = torch.tensor(0).double().to(device)

    for i in range(len(stats)):
        # If n+1 state is actually the last state
        if len(stats[i]) == 1:
            continue

        probs, values, rewards = lists_from_stats(stats[i])

        # Last reward is value of the last state. Clip the last return which the value of last state
        returns = get_returns(rewards + [values[-1]], gamma_returns)[:-1]
        credits_ = get_credits(len(rewards), gamma_credits)

        loss_v = F.mse_loss(values[:-1], returns)
        loss_p = torch.mean(-credits_ * returns * probs)

        loss += +0.1 * loss_v + 1 * loss_p

    return loss / len(stats)


def advantage_ac_loss(
    stats: List[List[Dict[str, Union[torch.Tensor, float]]]], gamma_returns: float
) -> torch.Tensor:
    loss = torch.tensor(0).double().to(device)

    for i in range(len(stats)):
        # If n+1 state is actually the last state
        if len(stats[i]) == 1:
            continue

        probs, values, rewards = lists_from_stats(stats[i])

        # Last reward is value of the last state. Clip the last return which the value of last state
        returns = get_returns(rewards + [values[-1]], gamma_returns)[:-1]

        loss_v = F.mse_loss(values[:-1], returns)
        loss_p = torch.mean(-probs * (returns - values[:-1]))

        loss += 0.1 * loss_v + 1 * loss_p

    return loss / len(stats)


def lists_from_stats(stats):
    probs = torch.log(
        torch.stack([stat["policy"] for stat in stats if "policy" in stat])
    )
    values = torch.stack([stat["value"] for stat in stats if "value" in stat])
    rewards = [stat["reward"] for stat in stats if "reward" in stat]

    return probs, values, rewards


def get_returns(rewards, gamma_returns):
    total_t = len(rewards)
    returns = []
    prev_return = 0
    for t in range(total_t):
        prev_return = rewards[total_t - t - 1] + (gamma_returns * prev_return)
        returns.append(prev_return)
    return torch.tensor(returns).flip(0).double().to(device)


def get_credits(t: int, gamma_credits):
    return torch.pow(gamma_credits, torch.arange(t).float()).flip(0).double().to(device)


if __name__ == "__main__":
    main()
