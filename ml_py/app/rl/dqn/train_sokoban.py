import gym
import griddly
from griddly import gd
import matplotlib.pyplot as plt

if __name__ != "__main__":

    env = gym.make("GDY-Sokoban-v0")
    state = env.reset()
    print(state)
    print(env.action_space)
    print(env.observation_space)
    env.render()

    while True:
        state, reward, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            env.reset()
            env.render()


if __name__ == "__main__":
    env = gym.make(f"GDY-Sokoban-v0", global_observer_type=gd.ObserverType.SPRITE_2D)
    state = env.reset()
    print(state)
    print(env.action_space)
    print(env.observation_space)
    # env.render()
