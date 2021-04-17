from typing import Type
import torch
import wandb
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from datetime import datetime
from app.rl.env_recorder import EnvRecorder
from app.rl.envs.env_wrapper import EnvWrapper, BatchEnvWrapper
from app.rl.prioritized_replay import (
    PrioritizedReplay,
    state_action_reward_state_2_transform,
)
from settings import BASE_DIR
from app.rl.dqn.utils import sample_actions, reset_envs_that_took_too_long


def train_dqn_per(
    env_class: Type[EnvWrapper], model: nn.Module, config: DictConfig, name=None
):
    batch = BatchEnvWrapper(env_class, config.batch_size)
    batch.reset()
    wandb.init(
        name=f"{name}_{str(datetime.now().timestamp())[5:10]}",
        project=name or "testing_dqn",
        config=config,
        save_code=True,
        group=None,
        tags=None,  # List of string tags
        notes=None,  # longer description of run
        dir=BASE_DIR,
    )
    wandb.watch(model)
    replay = PrioritizedReplay(
        config.replay_size,
        config.replay_batch,
        transform=state_action_reward_state_2_transform,
    )
    env_recorder = EnvRecorder(config.env_record_freq, config.env_record_duration)
    cumulative_reward = 0
    cumulative_done = 0

    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    current_episodic_steps = torch.zeros((config.batch_size,))

    for step in range(config.steps):
        log = DictConfig({})
        log.step = step

        replay_batch = replay.get_batch()
        states = torch.cat((batch.get_state_batch(), replay_batch[0]), 0)

        q_pred = model(states)

        actions_live = sample_actions(
            q_pred[: config.batch_size],
            batch.get_legal_actions(),
            epsilon=config.epsilon_exploration,
        )

        # ============ Observe the reward && predict value of next state ==============

        _, rewards, done_list, _ = batch.step(actions_live)

        rewards = torch.tensor(rewards).float()
        done_list = torch.tensor(done_list, dtype=torch.int8)

        current_episodic_steps += done_list
        reset_envs_that_took_too_long(
            batch.envs, current_episodic_steps, config.max_steps
        )

        next_states = torch.cat((batch.get_state_batch(), replay_batch[3]), 0)
        actions = torch.cat((torch.tensor(actions_live), replay_batch[1]), 0)
        model.eval()
        with torch.no_grad():
            q_next = model(next_states)
        model.train()

        value_live = rewards + config.gamma_discount * torch.amax(
            q_next[: config.batch_size], 1
        )
        value_replay = replay_batch[2] + config.gamma_discount * torch.amax(
            q_next[config.batch_size :], 1
        )
        value = torch.cat((value_live, value_replay), 0)

        q_actions = q_pred[:, actions]

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

        env_recorder.record(step, batch.envs, wandb)

        wandb.log(log)
