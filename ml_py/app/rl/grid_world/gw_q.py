from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from app.rl.grid_world.gw_pg import GWPgModel
from gym_grid_world.envs.grid_world_env import GridWorldEnv
from settings import BASE_DIR, device

grid_size = 4
epsilon = 0.1
gamma = 0.9

lr = 0.01

mode = 'random'


def main():
    env = GridWorldEnv(size=grid_size, mode=mode)
    model = GWPgModel(size=grid_size, units=[50]).double().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter(
        f'{BASE_DIR}/app/rl/grid_world/runs/gw_q_LR{str(lr)[:7]}_{mode}_{int(datetime.now().timestamp())}')

    for epoch in range(2000):
        env.reset()
        step = 0
        losses = []
        rewards = []

        while not env.done and step < 50:
            y = model(model.convert_inputs([env]))

            if torch.rand(1) < epsilon:
                action = torch.randint(0, 4, (1,))
                qh = y[0][action]
            else:
                action = torch.argmax(y, 1)
                qh = y[0][action]

            _, reward, _, _ = env.step(int(action))

            with torch.no_grad():
                model.eval()
                q_next, _ = torch.max(model(model.convert_inputs([env]))[0], dim=0)
                model.train()

            q = reward + gamma * q_next

            loss = (qh - q) ** 2

            losses.append(loss.item())
            rewards.append(reward)

            optim.zero_grad()
            loss.backward()
            optim.step()

        writer.add_scalar('loss', np.mean(losses), global_step=epoch)
        writer.add_scalar('reward', np.mean(rewards), global_step=epoch)
        writer.add_scalar('episode_len', len(losses), global_step=epoch)
        print('.', end='')


main()
