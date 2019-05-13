import random


def random_policy(right_percent):
    return int(random.random() < right_percent)


def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1