from datetime import datetime
from typing import List

import hydra
import numpy as np
import optuna
import torch
from omegaconf import OmegaConf, DictConfig
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from gym_grid_world.envs import GridWorldEnv
from lib.nn_utils import save_model
from settings import BASE_DIR, device

CWD = f"{BASE_DIR}/app/rl/grid_world"


class GWPgModel(nn.Module):
    def __init__(self, size: int, units: List[int]):
        super().__init__()
        self.size = size

        self.first = nn.Sequential(
            nn.Conv2d(4, units[0], kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(0.3)
        )

        self.hidden = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(units[i], units[i + 1], kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )
                for i in range(len(units) - 1)
            ]
        )

        self.out = nn.Linear(self.size * self.size * units[-1], 4)

    def forward(self, x):
        x = self.first(x)
        for hidden in self.hidden:
            x = hidden(x)
        x = x.flatten(1)
        return self.out(x)

    @staticmethod
    def convert_inputs(envs):
        """
        Outputs a tensor of shape(batch, 4,4,4)
        """
        inputs = np.array([env.state for env in envs])
        return torch.tensor(inputs).double().to(device)


class GWPolicyGradTrainer:
    def __init__(self, **config):
        self.cfg = OmegaConf.create(config)

        self.model = (
            GWPgModel(
                self.cfg.grid_size, [self.cfg.units for _ in range(self.cfg.depth)]
            )
            .double()
            .to(device)
        )
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.writer = SummaryWriter(
            f"{CWD}/runs/gw_policy_grad_LR{str(self.cfg.lr)[:7]}_{self.cfg.depth}x{self.cfg.units}_{int(datetime.now().timestamp())}"
        )
        self.envs = [
            GridWorldEnv(size=self.cfg.grid_size, mode=self.cfg.env_mode)
            for _ in range(self.cfg.n_env)
        ]
        self.stats_e = []
        self.won = []
        self.current_episode = 1

        self.reset_episode()
        self.writer.add_graph(self.model, GWPgModel.convert_inputs(self.envs))

    def get_credits(self, t):
        credits = []
        prev_credit = 1
        for i in range(t):
            credits.append(prev_credit)
            prev_credit *= self.cfg.gamma_credits
        return torch.tensor(list(reversed(credits))).double().to(device)

    def get_returns(self, rewards):
        total_t = len(rewards)
        returns = []
        prev_return = 0
        for t in range(total_t):
            prev_return = rewards[total_t - t - 1] + (
                self.cfg.gamma_returns * prev_return
            )
            returns.append(prev_return)
        return torch.tensor(list(reversed(returns))).double().to(device)

    def reset_episode(self):

        [env.reset() for env in self.envs]
        self.stats_e = [[] for _ in self.envs]
        self.won = [None for _ in self.envs]

    def sample_action(self, probs):
        # Softmax
        tau = max((1 / (np.log(self.current_episode) * 5 + 0.0001)), 0.7)
        probs = F.gumbel_softmax(probs, tau=tau, dim=0)
        # probs = F.softmax(probs, dim=0)

        # Add noise
        # noise = torch.rand(len(probs)) * 0.1
        # probs = probs + noise

        # Subsample legal probs
        # legal_probs = probs[legal_idx]

        # Sample action idx
        # if len(legal_probs) == 0:
        #     return 0, 0, 0
        action = torch.multinomial(probs, 1)[0]

        return action, probs[action]

    def run_time_step(self, yh):
        for i in range(self.cfg.n_env):

            if self.envs[i].done:
                continue

            action, prob = self.sample_action(yh[i])
            _, reward, done, _ = self.envs[i].step(action)
            # envs[i].render()

            self.stats_e[i].append({"reward": reward, "prob": prob})
            self.won[i] = done and self.envs[i].won

    def learn(self):
        loss = torch.tensor(0).double().to(device)
        rewards_list = []
        for i in range(self.cfg.n_env):
            probs = [stat["prob"] for stat in self.stats_e[i]]
            if len(probs) == 0:
                continue
            probs = torch.stack(probs)
            rewards = [stat["reward"] for stat in self.stats_e[i]]
            returns = self.get_returns(rewards)
            credits = self.get_credits(len(rewards))

            loss += torch.sum(probs * credits * returns)
            rewards_list.append(np.mean(rewards))

        loss = -1 * loss / self.cfg.n_env

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        # print(f"loss: {loss}")
        self.writer.add_scalar(
            "Training loss", loss.item(), global_step=self.current_episode
        )
        self.writer.add_scalar(
            "Mean Rewards", np.mean(rewards_list), global_step=self.current_episode
        )

        # losses.append(loss.item())

    def run_episode(self):
        # Reset envs
        self.reset_episode()
        step = 0

        while not all([env.done for env in self.envs]) and step < self.cfg.max_steps:
            # Predict actions

            x = GWPgModel.convert_inputs(self.envs)
            yh = self.model(x)

            self.run_time_step(yh)
            step += 1

        if step == self.cfg.max_steps:
            for i in range(self.cfg.n_env):
                if not self.envs[i].done:
                    self.stats_e[i].append({"reward": -10, "prob": torch.tensor(0)})

        self.learn()


"""
Log the following:
For each episode:
  - Episode length (min, max, avg)
  - Win / lose
  - sum of rewards
  - loss
For every nth episodes, for each timestep:
  - Policy histogram
  - value
"""


def play(model, cfg):
    env = GridWorldEnv(cfg.grid_size, cfg.env_mode)
    env.reset()
    env.render()
    step = 0

    while not env.done and step < cfg.max_steps:
        x = GWPgModel.convert_inputs([env])
        yh = model(x)
        yh = F.softmax(yh, 1)
        action = yh[0].argmax(0)

        _, reward, done, _ = env.step(action)

        env.render()
        step += 1


def get_final_reward(trainer):
    reward = 0
    n = 0
    for stat in trainer.stats_e:
        reward += stat[-1]["reward"]
        n += 1
    return reward / (n + 0.00001)


def run_trainer(cfg: DictConfig, trail: optuna.Trial) -> float:
    # cfg.lr = trail.suggest_loguniform('lr', 0.00001, 0.1)
    # cfg.depth = trail.suggest_int('depth', 1, 4)
    # cfg.units = trail.suggest_int('units', 5, 500)

    trainer = GWPolicyGradTrainer(**dict(cfg))

    while trainer.current_episode <= cfg.total_episodes:
        trainer.run_episode()
        print(".", end="")
        trainer.current_episode += 1

    final_reward = get_final_reward(trainer)
    hparams = {key: cfg[key] for key in ["lr", "depth", "units"]}
    trainer.writer.add_hparams(hparams, {"final_reward": final_reward})
    trainer.writer.close()

    # play(trainer.model, cfg)

    save_model(trainer.model, CWD, "grid_world_pg")

    return final_reward


@hydra.main(config_name="config/pg")
def main(cfg: DictConfig) -> None:
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trail: run_trainer(cfg, trail), n_trials=1)
    print(f"{study.best_params=}")
    print(f"{study.best_value=}")


if __name__ == "__main__":
    main()

# python gw_pg.py --multirun lr=0.0001,0.001,0.01,0.1
