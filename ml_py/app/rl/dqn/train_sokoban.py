import gym
import griddly
import matplotlib.pyplot as plt

if __name__ == "__main__":

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
