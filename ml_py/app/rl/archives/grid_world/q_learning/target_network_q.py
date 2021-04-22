import torch
from gym_grid_world.envs import GridWorldEnv
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from copy import deepcopy

from app.rl.archives.grid_world.PrioritizedReplay import PrioritizedReplay
from app.rl.archives.grid_world.gw_pg import GWPgModel
from app.rl.archives.grid_world.utils import state_to_dict
from lib.nn_utils import save_model
from settings import BASE_DIR, device

CWD = f"{BASE_DIR}/app/rl/grid_world/q_learning"


def main_single_batch():

    # ============= INITIALIZE VARIABLES ===================

    grid_size = 4
    epsilon = 0.1
    gamma = 0.95
    n_episodes = 15000
    max_steps = 50
    replay_batch_size = 199
    max_buffer_size = 1000
    sync_freq = 250  # NEW
    lr = 0.001
    mode = "random"
    architecture = [50]

    writer = SummaryWriter(
        f"{CWD}/runs/gw_target_network_q_LR{str(lr)[:7]}_{mode}_{int(datetime.now().timestamp())}"
    )
    env = GridWorldEnv(size=grid_size, mode=mode)
    experiences = PrioritizedReplay(max_size=max_buffer_size)

    model = GWPgModel(size=grid_size, units=architecture).double().to(device)

    model2 = deepcopy(model)
    model2.load_state_dict(model.state_dict())
    model2.eval()

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    # ============== TRAINING LOOP ===========================

    for epoch in range(n_episodes):
        env.reset()
        step = 0

        while not env.done and step <= max_steps:
            # Store state for experience replay
            state = env.state

            # =============== Collect experiences =================

            envs = [env]

            exp_samples = experiences.sample(replay_batch_size)
            for exp in exp_samples:
                new_env = GridWorldEnv(size=grid_size, mode=mode)
                new_env.reset()
                new_env.state = state_to_dict(exp[1])
                envs.append(new_env)

            x = model.convert_inputs(envs)
            x = x + torch.rand_like(x) / 100

            # =======================================================

            y = model(x)

            rewards = []
            qhs = []
            for i in range(len(envs)):
                # =========== Epsilon Probability ==============

                use_rand = torch.rand(1)[0] < epsilon
                action = (
                    torch.randint(0, 4, (1,))[0] if use_rand else torch.argmax(y[i], 0)
                )
                qh = y[i][action]

                # ============ Observe the reward && predict value of next state ==============

                _, reward, _, _ = envs[i].step(int(action))
                rewards.append(reward)
                qhs.append(qh)

            if step >= max_steps:
                rewards[0] = -10
                env.done = True
            rewards = torch.tensor(rewards).double().to(device)
            qhs = torch.stack(qhs)

            x_next = model2.convert_inputs(envs)
            with torch.no_grad():
                y_next = model2(x_next)
            q_next, _ = torch.max(y_next, dim=1)

            q = rewards + gamma * q_next

            # =========== LEARN ===============

            loss = (qhs - q) ** 2

            loss = torch.mean(loss)
            optim.zero_grad()
            loss.backward()
            optim.step()

            step += 1
            global_step += 1

            if global_step % sync_freq == 0:
                model2.load_state_dict(model.state_dict())

            writer.add_scalar("loss", loss.item(), global_step=global_step)
            writer.add_scalar("reward", rewards[0].item(), global_step=global_step)
        # writer.add_scalar("episode_len", step, global_step=global_step)
        print(".", end="")

    save_model(model, CWD + "/..", "grid_world_q")


if __name__ == "__main__":
    main_single_batch()
