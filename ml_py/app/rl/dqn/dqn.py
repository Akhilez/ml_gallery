"""

What's desired?
- Use yaml files for hyperparameters
  - because it is easy to comment and add whatever notes you want in the yaml file
- Use batches of envs
  - Not all envs are meant to start at the same time and when one env is done,
    it will be reset independent of other envs in the batch.
- Run the algo in steps rather than epoch + batch
- Use env wrapper

Questions:
- How do we handle 2 player setup?

"""

from typing import Type

import torch
import wandb
from omegaconf import DictConfig
from torch import nn

from app.rl.dqn.env_wrapper import EnvWrapper
from settings import BASE_DIR


def train_dqn(env_class: Type[EnvWrapper], model: nn.Module, config: DictConfig):
    envs = [env_class() for _ in range(config.batch_size)]
    [env.reset() for env in envs]
    wandb.init(
        # name="",  # Name of the run
        project="testing_dqn",
        config=config,
        save_code=True,
        group=None,
        tags=None,  # List of string tags
        notes=None,  # longer description of run
        dir=BASE_DIR,
    )

    optim = torch.optim.Adam(model.parameters(), lr=config.lr)

    for step in range(config.steps):
        log = DictConfig({})
        log.step = step

        states = env_class.get_state_batch(envs)
        q_pred = model(states)

        actions = sample_actions(q_pred, epsilon=config.epsilon)

        print(step)
        wandb.log(log)


def sample_actions(q_values: torch.Tensor, epsilon: float):
    batch_size = len(q_values)
    exploit_actions = torch.argmax(q_values, 1)
    explore_actions = torch.randint(low=0, high=batch_size, size=(batch_size,))
    random_indices = torch.multinomial(
        torch.Tensor([epsilon, 1 - epsilon]), batch_size, replacement=True
    )
    explore_indices = random_indices == 0
    exploit_actions[explore_indices] = explore_actions[explore_indices]
    return exploit_actions
