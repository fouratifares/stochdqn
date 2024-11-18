import itertools
import random
from cmath import inf
import numpy as np
import gymnasium as gym


def smooth_rewards(rewards, window_size):
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if window_size > len(rewards):
        raise ValueError("window_size cannot be greater than the length of rewards.")
    cumsum_vec = np.cumsum(np.insert(rewards, 0, 0))
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size


def palette(num_colors):
    colors = []
    for _ in range(num_colors):
        # Generate random values for red, green, and blue components
        red = random.random()
        green = random.random()
        blue = random.random()
        # Create an RGB color tuple
        color = (red, green, blue)
        colors.append(color)
    return colors


def index_to_action(num_actions, num_levels, range_=1):
    # Create a list of all possible values for a single action
    action_values = [i / (num_levels - 1) * (range_ * 2) - range_ for i in range(num_levels)]

    # Generate all possible combinations of actions
    all_action_combinations = list(itertools.product(action_values, repeat=num_actions))

    # Create a dictionary to store the combinations
    act_dict = {}
    for i, combination in enumerate(all_action_combinations):
        act_dict[i] = combination

    return act_dict


def define_environment(setting='InvertedPendulum-v4', render=False):


    if setting == 'InvertedPendulum-v4':
        if render:
            env = gym.make('InvertedPendulum-v4', render_mode='human')
        else:
            env = gym.make('InvertedPendulum-v4')
        env.num_states = float(inf)
        env.num_actions = 1
        return env

    return "Unknown environment setting"
