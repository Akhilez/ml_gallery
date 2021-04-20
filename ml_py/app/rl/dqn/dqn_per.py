"""
This file adds Prioritized Experience Replay feature to the vanilla dqn algorithm.
How it works:
1. Before training starts, envs are stepped once with random actions
2. The info is stored in the memory with 0 as the loss (cuz idk)
3.
"""


from typing import Type
import numpy as np
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
    env_class: Type[EnvWrapper],
    model: nn.Module,
    config: DictConfig,
    project_name=None,
    run_name=None,
):
    batch = BatchEnvWrapper(env_class, config.batch_size)
    batch.reset()
    wandb.init(
        name=f"{run_name}_{str(datetime.now().timestamp())[5:10]}",
        project=project_name or "testing_dqn",
        config=config,
        save_code=True,
        group=None,
        tags=None,  # List of string tags
        notes=None,  # longer description of run
        dir=BASE_DIR,
    )
    wandb.watch(model)
    replay = PrioritizedReplay(
        buffer_size=config.replay_size,
        batch_size=config.replay_batch,
        transform=state_action_reward_state_2_transform,
    )
    env_recorder = EnvRecorder(config.env_record_freq, config.env_record_duration)
    cumulative_reward = 0
    cumulative_done = 0

    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    current_episodic_steps = torch.zeros((config.batch_size,))

    store_initial_replay(batch, replay)

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

        rewards, dones = transform_step_data(*batch.step(actions_live))

        current_episodic_steps += dones
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
        rewards_all = torch.cat((rewards, replay_batch[2]))

        q_actions = q_pred[range(len(q_pred)), actions]

        # =========== LEARN ===============

        loss = F.mse_loss(q_actions, value, reduction="none")
        replay.add_batch(loss, (states, actions, rewards_all, next_states))
        loss = torch.mean(loss)

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

        cumulative_done += dones.sum()  # number of dones
        log.cumulative_done = int(cumulative_done)

        cumulative_reward += mean_reward
        log.cumulative_reward = cumulative_reward

        env_recorder.record(step, batch.envs, wandb)

        wandb.log(log)


def store_initial_replay(batched_env, buffer):
    legal_actions = batched_env.get_legal_actions()
    actions = [np.random.choice(actions) for actions in legal_actions]
    states1 = batched_env.get_state_batch()
    _, rewards, dones, infos = batched_env.step(actions)
    states2 = batched_env.get_state_batch()
    buffer.add_batch(np.zeros(len(states1)), (states1, actions, rewards, states2))


def transform_step_data(state, rewards, dones, info):
    rewards = torch.tensor(rewards).float()
    dones = torch.tensor(dones, dtype=torch.int8)
    return rewards, dones
