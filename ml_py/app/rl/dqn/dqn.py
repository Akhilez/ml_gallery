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

from omegaconf import OmegaConf

from app.rl.dqn.env_wrapper import EnvWrapper

hp = OmegaConf.load("config_dqn.yaml")

envs = []


def train_dqn(Env: Type[EnvWrapper], config_path: str):
    pass
