import gym_super_mario_bros
import torch
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from omegaconf import DictConfig

from app.rl.icm.mario import MarioModel, MarioICM, ExperienceReplay
from app.rl.icm.modular_test import Compose, DataInit, ProcessModule, Loop

hp = {
    "steps": 1,
    "lr": 1e-4,
    "max_episode_len": 1000,
    # Env specific
    "min_progress": 15,
    "frames_per_state": 3,
    "action_repeats": 6,
    # Action sampler
    "epsilon_random": 0.1,  # Sample random action with epsilon probability
    "epsilon_greedy_switch": 1000,
    # loss creator
    "gamma_q": 0.85,
    "q_loss_weight": 0.01,
    "inverse_loss_weight": 0.5,
    "forward_loss_weight": 0.5,
    # ICM
    "use_extrinsic": True,
    "intrinsic_weight": 1.0,
    "extrinsic_weight": 1.0,
}


class VarInit(ProcessModule):
    required_keys = [{"hp": ["frames_per_state", "lr"]}]

    def run(self):
        self.q_model = MarioModel(self.hp.frames_per_state)
        self.icm_model = MarioICM(self.hp.frames_per_state)

        self.optim = torch.optim.Adam(
            list(self.q_model.parameters()) + list(self.icm_model.parameters()),
            lr=self.hp.lr,
        )

        self.replay = ExperienceReplay(buffer_size=500, batch_size=100)
        self.env = gym_super_mario_bros.make("SuperMarioBros-v0")
        self.env = JoypadSpace(self.env, COMPLEX_MOVEMENT)

        # Counters and stats
        self.last_x_pos = 0
        self.current_episode = 0
        self.global_step = 0
        self.current_step = 0
        self.cumulative_reward = 0

        self.ep_rewards = []


class StepsLoop(Loop):
    required_keys = [{"hp": ["steps"]}, "global_step"]

    def termination_fn(self):
        return self.global_step >= self.hp.steps


graph = Compose(
    DataInit(hp),
    VarInit(),
    StepsLoop(
        DataInit(),
    ),
)
