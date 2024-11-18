import csv
import numpy as np
from src.utils import smooth_rewards
import matplotlib.pyplot as plt


def plot_(list_rewards, label='DQN', color='blue', linestyle='--', linewidth=0.8, alpha=0.1, window_size=10,
          x_axis=None, save_path=None):
    mean_total_rewards = np.mean(np.stack(list_rewards), axis=0)
    std_total_rewards = np.std(np.stack(list_rewards), axis=0)

    smoothed_mean_total_rewards = smooth_rewards(mean_total_rewards, window_size=window_size)
    smoothed_std_total_rewards = smooth_rewards(std_total_rewards, window_size=window_size)

    if x_axis is None:
        x_axis = list(range(len(smoothed_mean_total_rewards)))

    if save_path:
        # Save the mean and std total rewards to a CSV file
        with open(save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Step", "Mean Total Reward", "Std Total Reward"])
            for step, mean_reward, std_reward in zip(x_axis, smoothed_mean_total_rewards, smoothed_std_total_rewards):
                writer.writerow([step, mean_reward, std_reward])

    plt.plot(x_axis, smoothed_mean_total_rewards, label=label, color=color, linestyle=linestyle,
             linewidth=linewidth)
    plt.fill_between(x_axis, smoothed_mean_total_rewards - smoothed_std_total_rewards,
                     smoothed_mean_total_rewards + smoothed_std_total_rewards, alpha=alpha, color=color)
