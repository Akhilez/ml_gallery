"""
DQN with:
- Prioritized Experience Replay
- Epsilon decay
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
from app.rl.envs.decay_functions import decay_functions
from app.rl.envs.env_wrapper import EnvWrapper, BatchEnvWrapper
from app.rl.prioritized_replay import (
    PrioritizedReplay,
    state_action_reward_state_2_transform,
)
from settings import BASE_DIR
from app.rl.dqn.action_sampler import EpsilonRandomActionSampler


def train_dqn_e_decay(
    env_class: Type[EnvWrapper],
    model: nn.Module,
    config: DictConfig,
    project_name=None,
    run_name=None,
):
    env = BatchEnvWrapper(env_class, config.batch_size)
    env.reset()
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    epsilon_scheduler = decay_functions[config.epsilon_decay_function]
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
    replay = PrioritizedReplay(
        buffer_size=config.replay_size,
        batch_size=config.replay_batch,
        delete_freq=config.delete_freq,
        delete_percentage=config.delete_percentage,
        transform=state_action_reward_state_2_transform,
    )
    env_recorder = EnvRecorder(config.env_record_freq, config.env_record_duration)
    sample_actions = EpsilonRandomActionSampler()

    cumulative_reward = 0
    cumulative_done = 0

    # ======= Start training ==========

    # We need _some_ initial replay buffer to start with.
    store_initial_replay(env, replay)

    for step in range(config.steps):
        log = DictConfig({"step": step})

        (
            states_replay,
            actions_replay,
            rewards_replay,
            states2_replay,
        ) = replay.get_batch()
        states = _combine(env.get_state_batch(), states_replay)

        q_pred = model(states)

        epsilon_exploration = epsilon_scheduler(config, log)
        actions_live = sample_actions(
            valid_actions=env.get_legal_actions(),
            q_values=q_pred[: config.batch_size],
            epsilon=epsilon_exploration,
        )

        # ============ Observe the reward && predict value of next state ==============

        states2, actions, rewards, dones_live = step_with_replay(
            env, actions_live, actions_replay, states2_replay, rewards_replay
        )

        model.eval()
        with torch.no_grad():
            q_next = model(states2)
        model.train()

        # Bellman equation
        value = rewards + config.gamma_discount * torch.amax(q_next, 1)

        q_select_actions = q_pred[range(len(q_pred)), actions]

        # =========== LEARN ===============

        loss = F.mse_loss(q_select_actions, value, reduction="none")

        replay.add_batch(loss, (states, actions, rewards, states2))
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

        cumulative_done += dones_live.sum()  # number of dones
        log.cumulative_done = int(cumulative_done)

        cumulative_reward += mean_reward
        log.cumulative_reward = cumulative_reward

        log.epsilon_exploration = epsilon_exploration

        env_recorder.record(step, env.envs, wandb)

        wandb.log(log)


def store_initial_replay(batched_env, buffer):
    legal_actions = batched_env.get_legal_actions()
    actions = [np.random.choice(actions) for actions in legal_actions]
    states1 = batched_env.get_state_batch()

    _, rewards, dones, infos = batched_env.step(actions)

    states2 = batched_env.get_state_batch()
    buffer.add_batch(np.zeros(len(states1)), (states1, actions, rewards, states2))


def step_with_replay(env, actions_live, actions_replay, states2_replay, rewards_replay):
    rewards_live, dones_live = transform_step_data(*env.step(actions_live))

    states2 = _combine(env.get_state_batch(), states2_replay)
    actions = _combine(actions_live, actions_replay)
    rewards = _combine(rewards_live, rewards_replay)

    return states2, actions, rewards, dones_live


def transform_step_data(state, rewards, dones, info):
    rewards = torch.tensor(rewards).float()
    dones = torch.tensor(dones, dtype=torch.int8)
    return rewards, dones


def _combine(live_data, replay_data):
    return torch.cat((live_data, replay_data), 0)
