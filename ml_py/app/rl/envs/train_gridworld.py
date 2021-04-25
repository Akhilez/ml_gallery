from omegaconf import DictConfig
from app.rl.dqn.dqn import train_dqn
from app.rl.dqn.dqn_double import train_dqn_double
from app.rl.dqn.dqn_e_decay import train_dqn_e_decay
from app.rl.dqn.dqn_per import train_dqn_per
from app.rl.dqn.dqn_target import train_dqn_target
from app.rl.dqn.pg import train_pg
from app.rl.envs import decay_functions
from app.rl.envs.env_wrapper import GymEnvWrapper, NumpyStateMixin, TimeOutLostMixin
from app.rl.models import GenericConvModel
from gym_grid_world.envs import GridWorldEnv
from utils import device


class GridWorldEnvWrapper(TimeOutLostMixin, NumpyStateMixin, GymEnvWrapper):
    reward_range = (-10, 10)
    max_steps = 50

    def __init__(self):
        super().__init__(GridWorldEnv(size=4, mode="random"))

    def get_legal_actions(self):
        return self.env.get_legal_actions()


def dqn_gridworld():

    hp = DictConfig({})

    hp.steps = 1000
    hp.batch_size = 600
    hp.env_record_freq = 100
    hp.env_record_duration = 25

    hp.max_steps = 50
    hp.grid_size = 4

    hp.lr = 1e-3
    hp.epsilon_exploration = 0.1
    hp.gamma_discount = 0.9

    model = (
        GenericConvModel(height=4, width=4, in_channels=4, channels=[50], out_size=4)
        .float()
        .to(device)
    )

    train_dqn(
        GridWorldEnvWrapper, model, hp, project_name="SimpleGridWorld", run_name="dqn"
    )


def dqn_per_gridworld():
    hp = DictConfig({})

    hp.steps = 1000
    hp.batch_size = 500
    hp.replay_batch = 100
    hp.replay_size = 1000
    hp.delete_freq = 100 * (hp.batch_size + hp.replay_size)  # every 100 steps

    hp.env_record_freq = 100
    hp.env_record_duration = 25

    hp.max_steps = 50
    hp.grid_size = 4

    hp.lr = 1e-3
    hp.epsilon_exploration = 0.1
    hp.gamma_discount = 0.9

    model = (
        GenericConvModel(height=4, width=4, in_channels=4, channels=[50], out_size=4)
        .float()
        .to(device)
    )

    train_dqn_per(
        GridWorldEnvWrapper,
        model,
        hp,
        project_name="SimpleGridWorld",
        run_name="dqn_per",
    )


def dqn_e_decay_gw():
    hp = DictConfig({})

    hp.steps = 1000
    hp.batch_size = 500

    hp.replay_batch = 100
    hp.replay_size = 1000

    hp.delete_freq = 100 * (hp.batch_size + hp.replay_size)  # every 100 steps

    hp.env_record_freq = 100
    hp.env_record_duration = 25

    hp.max_steps = 50
    hp.grid_size = 4

    hp.lr = 1e-3
    hp.gamma_discount = 0.9

    # hp.epsilon_exploration = 0.1
    hp.epsilon_flatten_step = 700
    hp.epsilon_start = 1
    hp.epsilon_end = 0.001
    hp.epsilon_decay_function = decay_functions.LINEAR

    model = (
        GenericConvModel(height=4, width=4, in_channels=4, channels=[50], out_size=4)
        .float()
        .to(device)
    )

    train_dqn_e_decay(
        GridWorldEnvWrapper,
        model,
        hp,
        project_name="SimpleGridWorld",
        run_name="dqn_e_decay",
    )


def dqn_target():
    hp = DictConfig({})

    hp.steps = 1000
    hp.batch_size = 500

    hp.replay_batch = 100
    hp.replay_size = 1000

    hp.delete_freq = 100 * (hp.batch_size + hp.replay_size)  # every 100 steps

    hp.env_record_freq = 100
    hp.env_record_duration = 25

    hp.max_steps = 50
    hp.grid_size = 4

    hp.lr = 1e-3
    hp.gamma_discount = 0.9

    # hp.epsilon_exploration = 0.1
    hp.epsilon_flatten_step = 700
    hp.epsilon_start = 1
    hp.epsilon_end = 0.001
    hp.epsilon_decay_function = decay_functions.LINEAR

    hp.target_model_sync_freq = 50

    model = (
        GenericConvModel(height=4, width=4, in_channels=4, channels=[50], out_size=4)
        .float()
        .to(device)
    )

    train_dqn_target(
        GridWorldEnvWrapper,
        model,
        hp,
        project_name="SimpleGridWorld",
        run_name="dqn_target",
    )


def dqn_double():
    hp = DictConfig({})

    hp.steps = 1000
    hp.batch_size = 500

    hp.replay_batch = 100
    hp.replay_size = 1000

    hp.delete_freq = 100 * (hp.batch_size + hp.replay_size)  # every 100 steps

    hp.env_record_freq = 100
    hp.env_record_duration = 25

    hp.max_steps = 50
    hp.grid_size = 4

    hp.lr = 1e-3
    hp.gamma_discount = 0.9

    # hp.epsilon_exploration = 0.1
    hp.epsilon_flatten_step = 700
    hp.epsilon_start = 1
    hp.epsilon_end = 0.001
    hp.epsilon_decay_function = decay_functions.LINEAR

    hp.target_model_sync_freq = 50

    model = (
        GenericConvModel(height=4, width=4, in_channels=4, channels=[50], out_size=4)
        .float()
        .to(device)
    )

    train_dqn_double(
        GridWorldEnvWrapper,
        model,
        hp,
        project_name="SimpleGridWorld",
        run_name="dqn_target",
    )


def pg_gridworld():

    hp = DictConfig({})

    hp.episodes = 2
    hp.batch_size = 2

    hp.lr = 1e-3

    hp.gamma_discount_credits = 0.9
    hp.gamma_discount_returns = 0.9

    model = (
        GenericConvModel(height=4, width=4, in_channels=4, channels=[50], out_size=4)
        .float()
        .to(device)
    )

    train_pg(
        GridWorldEnvWrapper, model, hp, project_name="SimpleGridWorld", run_name="pg"
    )


if __name__ == "__main__":
    pg_gridworld()
