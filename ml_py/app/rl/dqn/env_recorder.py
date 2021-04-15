from typing import List
import matplotlib.pyplot as plt
import numpy as np

from app.rl.dqn.env_wrapper import EnvWrapper


class EnvRecorder:
    def __init__(self, frequency: int = 0, duration: int = 0, n_envs: int = 1):
        self.frequency = frequency
        self.duration = duration
        self.n_envs = n_envs
        assert n_envs > 0
        self.buffer = [[] for _ in range(n_envs)]

    def record(self, step: int, envs: List[EnvWrapper], wandb_run):
        if self.frequency and step % self.frequency < self.duration:
            for i in range(self.n_envs):
                self.buffer[i].append(envs[i].render("rgb_array"))
            if len(self.buffer[0]) >= self.duration:
                wandb_run.log(
                    {
                        f"video_{step}_{i}": wandb_run.Video(
                            self._format_video(self.buffer[i]), fps=4, format="gif"
                        )
                        for i in range(self.n_envs)
                    },
                    commit=False,
                )
                self.buffer = [[] for _ in range(self.n_envs)]

    @staticmethod
    def _format_video(video):
        video = np.array(video)
        video = np.moveaxis(video, -1, 1)
        return video
