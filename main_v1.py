import random

import gym
import mlflow
import numpy as np


def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1


def main():
    env = gym.make("CartPole-v0")
    n_episodes = 500
    max_steps = 1000
    totals = []

    for episode in range(n_episodes):
        ep_reward = 0
        obs = env.reset()
        for step in range(max_steps):
            action = basic_policy(obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if done:
                break
        totals.append(ep_reward)
    with mlflow.start_run(experiment_id=1):
        mlflow.log_metric("mean reward", np.mean(totals))
        mlflow.log_metric("std reward", np.std(totals))


if __name__ == '__main__':
    main()
