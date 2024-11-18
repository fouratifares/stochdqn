import os
import random
import numpy as np
from src.deep_experiments import deep_experiment, deep_test
from src.utils import define_environment
from src.hyperparameters import hyperparameters

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    hypers = hyperparameters()

    random.seed(hypers['seed'])
    np.random.seed(hypers['seed'])

    # Define the environment
    env = define_environment(setting=hypers['setting'], render=hypers['render'])
    env.reset()

    path = hypers['setting'] + "_discretize_" + str(hypers['discretize_step'])

    output_directory = 'figures/' + path
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Specify the file path where you want to save the dictionary
    file_path = output_directory + '/my_parameters.txt'

    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Loop through the dictionary and write each key-value pair to the file
        for key, value in hypers.items():
            file.write(f'{key}: {value}\n')

    # Prints
    print('Problem with Continuous Actions')
    print('environment', hypers['setting'])
    print('dim actions', hypers['dim_actions'])
    total_actions = hypers['discretize_step'] ** hypers['dim_actions']
    print('Discritize the Action Space into :', total_actions, ' actions')

    if hypers['training']:
        deep_experiment(env, hypers['num_episodes'], hypers['stopping_time'], output_directory, hypers['agent_types'], hypers['stoch'], hypers['reps'], hypers['setting'], hypers['window_size'],
                        hypers['discretize_step'])

    else:
        # Test the trained agent over max_steps!
        max_steps = 100
        while True:
            env = define_environment(setting=hypers['setting'], render=True)
            total_reward = deep_test(env, output_directory, max_steps, hypers['discretize_step'],
                                     agent_type='StochDQN')
            print(total_reward)
