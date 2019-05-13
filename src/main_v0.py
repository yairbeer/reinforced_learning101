import gym
import mlflow
import numpy as np

from src.policies import random_policy


def main():
    env = gym.make("CartPole-v0")
    n_episodes = 500
    max_steps = 1000
    totals = []

    for right_percent in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for episode in range(n_episodes):
            ep_reward = 0
            _ = env.reset()
            for step in range(max_steps):
                action = random_policy(right_percent)
                _, reward, done, info = env.step(action)
                ep_reward += reward
                if done:
                    break
            totals.append(ep_reward)
        with mlflow.start_run():
            mlflow.log_param("right percent", right_percent)
            mlflow.log_metric("mean reward", np.mean(totals))
            mlflow.log_metric("std reward", np.std(totals))


if __name__ == '__main__':
    main()
