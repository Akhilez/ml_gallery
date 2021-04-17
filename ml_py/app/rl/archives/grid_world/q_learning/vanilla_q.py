import torch
from gym_grid_world.envs import GridWorldEnv
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from app.rl.grid_world.gw_pg import GWPgModel
from settings import BASE_DIR, device

CWD = f"{BASE_DIR}/app/rl/grid_world/q_learning"


def main_single_batch():

    # ============= INITIALIZE VARIABLES ===================

    grid_size = 4
    epsilon = 0.1
    gamma = 0.9

    n_episodes = 10000
    max_steps = 150

    lr = 0.01

    mode = "player"

    writer = SummaryWriter(
        f"{CWD}/runs/gw_vanilla_q_LR{str(lr)[:7]}_{mode}_{int(datetime.now().timestamp())}"
    )

    env = GridWorldEnv(size=grid_size, mode=mode)

    model = GWPgModel(size=grid_size, units=[10]).double().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # ============== TRAINING LOOP ===========================

    for epoch in range(n_episodes):
        env.reset()
        step = 0
        losses = []
        rewards = []

        # Monte Carlo loop
        while not env.done and step < max_steps:
            x = model.convert_inputs([env])
            y = model(x)

            # =========== Epsilon Probability ==============

            if torch.rand(1) < epsilon:
                action = torch.randint(0, 4, (1,))
                qh = y[0][action]
            else:
                action = torch.argmax(y, 1)
                qh = y[0][action]

            # ============ Observe the reward && predict value of next state ==============

            _, reward, _, _ = env.step(int(action))

            with torch.no_grad():
                model.eval()
                q_next, _ = torch.max(model(model.convert_inputs([env]))[0], dim=0)
                model.train()

            q = reward + gamma * q_next

            # =========== LEARN ===============

            loss = (qh - q) ** 2

            losses.append(loss.item())
            rewards.append(reward)

            optim.zero_grad()
            loss.backward()
            optim.step()

            step += 1

        writer.add_scalar("loss", np.mean(losses), global_step=epoch)
        writer.add_scalar("reward", np.mean(rewards), global_step=epoch)
        writer.add_scalar("episode_len", len(losses), global_step=epoch)
        print(".", end="")

    # save_model(model, CWD, "grid_world_q")


if __name__ == "__main__":
    main_single_batch()
