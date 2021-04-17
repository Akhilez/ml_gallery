from datetime import datetime
from typing import List

import hydra
import numpy as np
import optuna
import torch
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from app.rl.grid_world.PrioritizedReplay import PrioritizedReplay
from app.rl.grid_world.gw_pg import GWPgModel
from gym_grid_world.envs.grid_world_env import GridWorldEnv
from lib.nn_utils import save_model
from settings import BASE_DIR, device


config: DictConfig = None
writer: SummaryWriter = None
model: torch.nn.Module = None
optim: torch.optim.Adam = None
envs: List = []
stats_e = []
replay = PrioritizedReplay(max_size=10)
losses = []
rewards = []
current_episode = 1

CWD = f"{BASE_DIR}/app/rl/grid_world"


def init():
    global config, current_episode, writer, model, optim, envs

    writer = SummaryWriter(
        f"{CWD}/runs/gw_q_LR{str(config.lr)[:7]}_{config.env_mode}_{int(datetime.now().timestamp())}"
    )

    units = [int(config.units) for _ in range(int(config.depth))]
    model = GWPgModel(size=config.grid_size, units=units).double().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    envs = [
        GridWorldEnv(size=config.grid_size, mode=config.env_mode)
        for _ in range(config.n_env)
    ]
    [env.reset() for env in envs]

    writer.add_graph(model, GWPgModel.convert_inputs(envs))


def main_single_batch():
    grid_size = 4
    epsilon = 0.1
    gamma = 0.9

    n_episodes = 10000
    max_steps = 50

    lr = 0.01

    mode = "random"

    env = GridWorldEnv(size=grid_size, mode=mode)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_episodes):
        env.reset()
        step = 0
        losses = []
        rewards = []

        while not env.done and step < max_steps:
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

            step += 1

        writer.add_scalar("loss", np.mean(losses), global_step=epoch)
        writer.add_scalar("reward", np.mean(rewards), global_step=epoch)
        writer.add_scalar("episode_len", len(losses), global_step=epoch)
        print(".", end="")

    save_model(model, CWD, "grid_world_q")


def calculate_epsilon():
    max_episode = 500

    # A number between 0 and 1. Grows from 0 to 1
    decay = min(max_episode, current_episode) / min(max_episode, config.total_episodes)

    epsilon = max(0.1, 1 - decay)
    return epsilon


def sample_action(yi):
    epsilon = calculate_epsilon()
    action = (
        torch.randint(0, 4, (1,)) if torch.rand(1) < epsilon else torch.argmax(yi, 0)
    )
    q = yi[action]
    return action, q.view(1)


def learn():
    optim.zero_grad()

    with torch.no_grad():
        model.eval()
        x_next = model.convert_inputs(envs)
        x_next
        yh_next = model()
        q_next, _ = torch.max(yh_next, dim=1)
        model.train()

    reward = []
    qh = []
    q_next_ = []

    for i in range(config.n_env):
        if envs[i].done:
            continue
        reward.append(stats_e[i][-1]["reward"])
        qh.append(stats_e[i][-1]["q"])
        q_next_.append(q_next[i])

    if len(qh) == 0:
        return

    qh = torch.stack(qh)
    reward = torch.tensor(reward)
    q_next = torch.stack(q_next_)

    q = reward + config.gamma * q_next

    loss = (qh - q) ** 2

    losses.append(loss.item())
    rewards.append(torch.sum(reward).item())

    loss.backward()
    optim.step()


def run_time_step(yh):
    for i in range(config.n_env):

        if envs[i].done:
            continue

        action, q = sample_action(yh[i])
        _, reward, done, _ = envs[i].step(action)
        # envs[i].render()

        stats_e[i].append({"reward": reward, "q": q})

    learn()


def reset_episode():
    global stats_e, losses, rewards

    [env.reset() for env in envs]
    stats_e = [[] for _ in envs]
    losses = []
    rewards = []


def run_episode():
    # Reset envs
    reset_episode()
    step = 0

    while not all([env.done for env in envs]) and step < config.max_steps:
        # Predict actions

        x = GWPgModel.convert_inputs(envs)
        # TODO: Implement experience appending
        # exp = gather_experiences()
        # xs = torch.cat(x, exp)
        yh = model(x)

        run_time_step(yh)
        step += 1

    if step == config.max_steps:
        for i in range(config.n_env):
            if not envs[i].done:
                stats_e[i].append({"reward": -10, "prob": torch.tensor(0)})

    writer.add_scalar("loss", np.mean(losses), global_step=current_episode)
    writer.add_scalar("reward", np.mean(rewards), global_step=current_episode)
    writer.add_scalar("episode_len", len(losses), global_step=current_episode)


def get_final_reward() -> float:
    return 0


def run_trainer(cfg: DictConfig, trail: optuna.Trial) -> float:
    global config, current_episode

    # cfg.lr = trail.suggest_loguniform('lr', 0.001, 0.1)
    # cfg.gamma = trail.suggest_uniform('gamma', 0.5, 0.99)

    config = cfg

    init()

    while current_episode <= cfg.total_episodes:
        run_episode()
        print(".", end="")
        current_episode += 1

    final_reward = get_final_reward()
    hparams = {key: cfg[key] for key in ["lr", "gamma"]}
    writer.add_hparams(hparams, {"final_reward": final_reward})
    writer.close()

    # play(trainer.model, cfg)

    save_model(model, CWD, "grid_world_pg")

    return final_reward


@hydra.main(config_name="config/q")
def main(cfg: DictConfig) -> None:
    run_trainer(cfg, None)
    # study = optuna.create_study(direction='maximize')
    # study.optimize(lambda trail: run_trainer(cfg, trail), n_trials=20)
    # print(f'{study.best_params=}')
    # print(f'{study.best_value=}')


if __name__ == "__main__":
    main()

# python gw_q.py --multirun lr=0.0001,0.001,0.01,0.1
# TODO: Bring Prioritized experience replay here
