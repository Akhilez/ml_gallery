from app.rl.dqn.dqn import train_dqn
from app.rl.dqn.env_wrapper import GymEnvWrapper
from gym_grid_world.envs import GridWorldEnv


class GridWorldEnvWrapper(GymEnvWrapper):
    def __init__(self):
        super().__init__()
        self.env = GridWorldEnv(size=4, mode="random")

    def get_legal_actions(self):
        return self.env.get_legal_actions()


train_dqn(GridWorldEnvWrapper, "config_dqn.yaml")
