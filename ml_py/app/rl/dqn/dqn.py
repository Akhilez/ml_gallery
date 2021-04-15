from typing import Type, List, Tuple
import torch
import wandb
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from datetime import datetime
from app.rl.dqn.env_recorder import EnvRecorder
from app.rl.dqn.env_wrapper import EnvWrapper, BatchEnvWrapper
from settings import BASE_DIR


def train_dqn(
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
    env_recorder = EnvRecorder(config.env_record_freq, config.env_record_duration)
    cumulative_reward = 0
    cumulative_done = 0

    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    current_episodic_steps = torch.zeros((config.batch_size,))

    for step in range(config.steps):
        log = DictConfig({})
        log.step = step

        states = batch.get_state_batch()
        q_pred = model(states)

        actions = sample_actions(
            q_pred, batch.get_legal_actions(), epsilon=config.epsilon_exploration
        )

        # ============ Observe the reward && predict value of next state ==============

        _, rewards, done_list, _ = batch.step(actions)

        rewards = torch.tensor(rewards).float()
        done_list = torch.tensor(done_list, dtype=torch.int8)

        current_episodic_steps += done_list
        reset_envs_that_took_too_long(
            batch.envs, current_episodic_steps, config.max_steps
        )

        next_states = batch.get_state_batch()
        model.eval()
        with torch.no_grad():
            q_next = model(next_states)
        model.train()

        value = rewards + config.gamma_discount * torch.amax(q_next, 1)
        q_actions = q_pred[:, actions]

        # =========== LEARN ===============

        loss = F.mse_loss(q_actions, value)

        optim.zero_grad()
        loss.backward()
        optim.step()

        # ============ Logging =============

        log.loss = loss.item()

        cumulative_done += done_list.sum()  # number of dones
        log.cumulative_done = int(cumulative_done)

        max_reward = torch.amax(rewards, 0).item()
        log.max_reward = max_reward

        mean_reward = torch.mean(rewards, 0).item()
        log.mean_reward = mean_reward

        cumulative_reward += mean_reward
        log.cumulative_reward = cumulative_reward

        env_recorder.record(step, batch.envs, wandb)

        wandb.log(log)


def sample_actions(
    q_values: torch.Tensor, valid_actions: List[List[int]], epsilon: float
) -> List[int]:
    # q_values: tensor of shape (batch, num_actions)
    # Valid_actions: tensor of shape (batch, any)
    return [
        sample_action(q_values[i], valid_actions[i], epsilon)
        for i in range(len(q_values))
    ]


def sample_action(
    q_values: torch.Tensor, valid_actions: List[int], epsilon: float
) -> int:
    # q_values of shape (num_actions)
    # valid_actions of shape (any)

    if torch.rand((1,)) < epsilon:
        # explore
        action_index = torch.randint(low=0, high=len(valid_actions), size=(1,))
    else:
        action_index = q_values[valid_actions].argmax(0)

    return valid_actions[action_index]


def reset_envs_that_took_too_long(
    envs: List[EnvWrapper], steps: torch.Tensor, max_steps: int
):
    env_indices = torch.nonzero(steps >= max_steps)
    for index in env_indices:
        envs[index].reset()
