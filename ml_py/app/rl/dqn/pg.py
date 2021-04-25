"""
Vanilla Policy Gradients

- Wait for all envs to finish episode yo!

"""

from typing import Type, List
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from datetime import datetime
from app.rl.dqn.action_sampler import ProbabilityActionSampler
from app.rl.env_recorder import EnvRecorder
from app.rl.envs.env_wrapper import EnvWrapper, DoneIgnoreBatchedEnvWrapper
from settings import BASE_DIR


class Stats:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.rewards: List[List[float]] = [[] for _ in range(self.batch_size)]
        self.probs: List[List[torch.Tensor]] = [[] for _ in range(self.batch_size)]
        self.dones: List[bool] = [False for _ in range(self.batch_size)]
        self.end_steps = np.zeros(self.batch_size, dtype=int)

    def record(self, rewards, actions, p_pred, done_list):
        for env_index, (reward, action, probs, done) in enumerate(
            zip(rewards, actions, p_pred, done_list)
        ):
            prob = torch.tensor(0) if self.dones[env_index] else probs[action]

            self.rewards[env_index].append(reward)
            self.probs[env_index].append(prob)

            if done and not self.dones[env_index]:
                self.end_steps[env_index] = len(self.rewards[env_index])
                self.dones[env_index] = True

    def get_returns(self, gamma: float):
        returns = torch.tensor(np.zeros_like(self.rewards), dtype=torch.float32)
        for i in range(self.batch_size):
            rewards = self.rewards[i]
            end_step = self.end_steps[i]
            discount = 1.0
            return_ = 0.0
            for step in range(end_step - 1, -1, -1):
                return_ = rewards[step] + discount * return_
                discount *= gamma
                returns[i][step] = return_
        return returns

    def get_credits(self, gamma: float):
        credits = torch.tensor(np.zeros_like(self.rewards), dtype=torch.float32)
        batch_size, steps = credits.shape
        discounts = gamma ** torch.arange(steps)  # [1, 0.9, 0.8, ...]
        discounts = reversed(discounts)

        for i in range(batch_size):
            end_step = self.end_steps[i]
            credits[i, :end_step] = discounts[steps - end_step :]

            # for step in range(end_step - 1, -1, -1):
            #     credits[i][step] = discounts[end_step - step - 1]

        return credits

    def get_probs(self):
        probs = [torch.stack(prob) for prob in self.probs]
        probs = torch.stack(probs).squeeze()
        return probs

    def get_mean_rewards(self):
        rewards = []
        for i in range(self.batch_size):
            end_step = self.end_steps[i]
            rewards.append(np.mean(self.rewards[i][:end_step]))
        return float(np.mean(rewards))


def train_pg(
    env_class: Type[EnvWrapper],
    model: nn.Module,
    config: DictConfig,
    project_name=None,
    run_name=None,
):
    env = DoneIgnoreBatchedEnvWrapper(env_class, config.batch_size)
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
    # TODO: Episodic env recorder?
    env_recorder = EnvRecorder(config.env_record_freq, config.env_record_duration)
    sample_actions = ProbabilityActionSampler()

    cumulative_reward = 0
    cumulative_done = 0
    stats = []

    # ======= Start training ==========

    for episode in range(config.episodes):
        stats = Stats(config.batch_size)  # Stores (reward, policy prob)
        step = 0
        env.reset()

        # Monte Carlo loop
        while not env.is_done("all"):
            log = DictConfig({"step": step})

            states = env.get_state_batch()
            p_pred = model(states)
            p_pred = F.softmax(p_pred, 1)

            actions = sample_actions(
                valid_actions=env.get_legal_actions(), probs=p_pred, noise=0.1
            )

            _, rewards, done_list, _ = env.step(actions)

            stats.record(rewards, actions, p_pred, done_list)

            # ======== Step logging =========

            mean_reward = float(np.mean(rewards))
            log.mean_reward = mean_reward

            cumulative_done += mean_reward
            log.cumulative_reward = cumulative_reward

            cumulative_done += float(np.sum(done_list))
            log.cumulative_done = cumulative_done

            # TODO: Log policy histograms

            wandb.log(log)

            step += 1

        returns = stats.get_returns(config.gamma_discount_returns)
        credits = stats.get_credits(config.gamma_discount_credits)
        probs = stats.get_probs()

        loss = -1 * (probs * credits * returns)
        loss = torch.sum(loss)

        optim.zero_grad()
        loss.backward()
        optim.step()

        # ======== Episodic logging ========

        log = DictConfig({"episode": episode})
        log.episodic_reward = stats.get_mean_rewards()

        wandb.log(log)
