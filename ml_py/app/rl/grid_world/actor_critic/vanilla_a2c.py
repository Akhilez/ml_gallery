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
    EPOCHS = 1
    BATCH_SIZE = 2
    MAX_MONTE_CARLO_STEPS = 50
    N_TRAIN_STEP = 4
    ARCHITECTURE = [50, 50]
    GAMMA_RETURNS = 0.80
    GAMMA_CREDITS = 0.95
    LEARNING_RATE = 1e-3

    # -------------- Setup other variables ----------------

    model = GwAcModel(GRID_SIZE, ARCHITECTURE).double().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    envs = [GridWorldEnv(size=GRID_SIZE, mode=ENV_MODE) for _ in range(BATCH_SIZE)]
    step = 0

    # -------------- Training loop ----------------
    for epoch in range(EPOCHS):

        [env.reset() for env in envs]
        stats = [[] for _ in envs]

        # -------------- Monte Carlo Loop ---------------------
        while True:

            # ----------- Predict policy and value -------------
            states = GwAcModel.convert_inputs(envs)
            ph, vh = model(states)  # Shapes: ph: (batch, 4); vh: (batch, 1)

            # ------------ Sample actions -----------------
            tau = max((1 / (np.log(epoch) * 5 + 0.0001)), 0.7)
            ph = F.gumbel_softmax(ph, tau=tau, dim=1)
            actions = torch.multinomial(ph, len(ph)).squeeze()  # shape: (batch)

            # ------------- Rewards from step ----------------
            for i in range(BATCH_SIZE):
                if not envs[i].done:
                    _, reward, _, _ = envs[i].step(actions[i])
                    stats[i].append({"reward": reward, "value": vh[i], "policy": ph[actions[i]]})

            step += 1

            # -------------- Termination conditions ------------------

            all_done = all([env.done for env in envs])
            has_timed_out = step < MAX_MONTE_CARLO_STEPS
            n_step_ended = step % N_TRAIN_STEP == 0

            if has_timed_out:
                # Set unfinished env's reward to -10
                for i in range(BATCH_SIZE):
                    if not envs[i].done:
                        stats[i][-1]['reward'] = -10
                all_done = True

            if all_done or n_step_ended:
                # TODO: Train
                pass

            if all_done:
                break


if __name__ == "__main__":
    main()
