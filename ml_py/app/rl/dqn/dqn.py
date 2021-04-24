from typing import Type
import torch
import wandb
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from datetime import datetime

from app.rl.dqn.action_sampler import EpsilonRandomActionSampler
from app.rl.env_recorder import EnvRecorder
from app.rl.envs.env_wrapper import EnvWrapper, BatchEnvWrapper
from settings import BASE_DIR


def train_dqn(
    env_class: Type[EnvWrapper],
    model: nn.Module,
    config: DictConfig,
    project_name=None,
    run_name=None,
):
    env = BatchEnvWrapper(env_class, config.batch_size)
    env.reset()
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    wandb.init(
        name=f"{run_name}_{str(datetime.now().timestamp())[5:10]}",
        project=project_name or "testing_dqn",
        config=dict(config),
        save_code=True,
        group=None,
        tags=None,  # List of string tags
        notes=None,  # longer description of run
        dir=BASE_DIR,
    )
    wandb.watch(model)
    env_recorder = EnvRecorder(config.env_record_freq, config.env_record_duration)
    sample_actions = EpsilonRandomActionSampler()

    cumulative_reward = 0
    cumulative_done = 0

    # ======= Start training ==========

    for step in range(config.steps):
        log = DictConfig({"step": step})

        states = env.get_state_batch()
        q_pred = model(states)

        actions = sample_actions(
            valid_actions=env.get_legal_actions(),
            q_values=q_pred,
            epsilon=config.epsilon_exploration,
        )

        # ============ Observe the reward && predict value of next state ==============

        _, rewards, done_list, _ = env.step(actions)

        rewards = torch.tensor(rewards).float()
        done_list = torch.tensor(done_list, dtype=torch.int8)
        next_states = env.get_state_batch()

        model.eval()
        with torch.no_grad():
            q_next = model(next_states)
        model.train()

        value = rewards + config.gamma_discount * torch.amax(q_next, 1)
        q_actions = q_pred[range(config.batch_size), actions]

        # =========== LEARN ===============

        loss = F.mse_loss(q_actions, value)

        optim.zero_grad()
        loss.backward()
        optim.step()

        # ============ Logging =============

        log.loss = loss.item()

        max_reward = torch.amax(rewards, 0).item()
        min_reward = torch.amin(rewards, 0).item()
        mean_reward = torch.mean(rewards, 0).item()
        log.max_reward = max_reward
        log.min_reward = min_reward
        log.mean_reward = mean_reward

        cumulative_done += done_list.sum()  # number of dones
        log.cumulative_done = int(cumulative_done)

        cumulative_reward += mean_reward
        log.cumulative_reward = cumulative_reward

        env_recorder.record(step, env.envs, wandb)

        wandb.log(log)
