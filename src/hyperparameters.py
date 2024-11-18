from cmath import inf
import numpy as np


def hyperparameters():
    hypers = {'seed': 42,
              'training': True,  # if false then test the trained agent
              'window_size': 2,
              'reps': 1,
              'render': False,  # human render
              'setting': 'InvertedPendulum-v4',
              'agent_types': ['DQN', 'Random'],  # Compare DQN family and Random
              'stoch': [True],  # True -> StochDQN and False -> Deterministic
              }

    if hypers['setting'] == 'InvertedPendulum-v4':
        hypers['n'] = np.log2(1)  # Number of base actions, we have 2**n possible actions.
        hypers['num_states'] = float(inf)
        hypers['num_episodes'] = 500
        hypers['stopping_time'] = 100
        hypers['dynamic_lr'] = True
        hypers['discretize_step'] = 1000  # We divide the action space into intervals

    hypers['dim_actions'] = int(2 ** hypers['n'])

    return hypers
