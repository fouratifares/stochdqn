import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from rl.stochdqn import StochDQNAgent
from src.plot import plot_
from src.utils import index_to_action, palette, define_environment


def deep_train(agent, env, num_episodes, setting, output_directory, max_steps,
               dictionary_index_to_action,
               agent_type, discretize_step):
    # Training loop
    test_as_after_episode = False

    total_rewards = []
    test_rewards = []
    for _ in tqdm(range(num_episodes)):
        state = env.reset()[0]
        total_reward = 0
        done = False
        steps = 0
        while not done:
            steps += 1

            random_actions = None
            if agent_type != 'DQN':
                random_actions = np.random.choice(agent.action_size, agent.log2_actions, replace=False)
                random_actions = torch.tensor(random_actions, dtype=torch.int64)

            if agent_type == 'Random':
                action = random_actions[0].item()
            else:
                action = agent.select_action(state, random_actions)

            action_ = dictionary_index_to_action[action]

            next_state, reward, done, truncated, _ = env.step(action_)

            if agent_type != 'Random':
                agent.train(state, action, reward, next_state, done, random_actions)

            state = next_state
            total_reward += reward

            if steps >= max_steps or truncated:
                done = True

            if agent_type != 'Random':
                agent.update_target_network()

        total_rewards.append(total_reward)
        print('total reward:', total_reward)

        # Save the trained model if needed
        torch.save(agent.model.state_dict(), output_directory + '/' + agent_type + '_model.pth')

        if test_as_after_episode and agent_type != 'Random':
            env = define_environment(num_states=env.observation_space.shape[0], num_actions=env.num_actions,
                                     setting=setting, render=False)
            test_reward = deep_test(env, output_directory, max_steps, discretize_step,
                                    agent_type=agent_type)

            # print('test reward:', test_reward)
            test_rewards.append(test_reward)

    env.close()
    return total_rewards, test_rewards


def deep_experiment(env, num_episodes, max_steps, output_directory, agent_types, deter, reps, setting, window_size,
                    discretize_step=10):
    state_size = env.observation_space.shape[0]

    action_size = discretize_step ** env.num_actions
    dictionary_index_to_action = index_to_action(env.num_actions, discretize_step)
    assert len(dictionary_index_to_action.keys()) == action_size

    print(dictionary_index_to_action[0])
    print(dictionary_index_to_action[action_size - 1])

    for agent_type in agent_types:
        if agent_type == 'DQN':
            deter = deter
            num_colors = len(deter)
            colors = palette(num_colors)
            print('colors: ', len(colors))
            i = 0
            for stoch in deter:
                color = colors[i]
                if not stoch:
                    label = agent_type
                else:
                    label = 'Stoch' + agent_type
                print(label)
                i += 1
                list_rewards = []
                list_test_rewards = []

                for _ in tqdm(range(reps)):
                    agent = StochDQNAgent(state_size, action_size, dictionary_index_to_action,
                                          learning_rate=0.001, gamma=0.99, epsilon=1,
                                          epsilon_decay=0.995, epsilon_min=0.01,
                                          deterministic=not stoch)
                    total_rewards, test_rewards = deep_train(agent, env, num_episodes, setting,
                                                             output_directory,
                                                             max_steps,
                                                             dictionary_index_to_action, label, discretize_step)

                    list_rewards.append(total_rewards)
                    list_test_rewards.append(test_rewards)

                print("list_rewards", list_rewards)
                plot_(list_rewards, label=label, color=color, linestyle='-', linewidth=0.8, alpha=0.1,
                      window_size=window_size, save_path=output_directory + '/rewards_' + label + '.csv')

        elif agent_type == 'Random':
            list_rewards = []

            for _ in tqdm(range(reps)):
                agent = StochDQNAgent(state_size, action_size, dictionary_index_to_action, hidden_size=64,
                                      learning_rate=0.001, gamma=0.99, epsilon=1.0,
                                      epsilon_decay=1, epsilon_min=1.0)
                total_rewards, test_rewards = deep_train(agent, env, num_episodes, setting, output_directory,
                                                         max_steps,
                                                         dictionary_index_to_action, 'Random', discretize_step)
                list_rewards.append(total_rewards)
            plot_(list_rewards, label=agent_type, color='gray', linestyle='-', linewidth=0.8, alpha=0.1,
                  window_size=window_size, save_path=output_directory + '/rewards_Random.csv')

    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.title('Average Total Rewards over Episodes for ' + setting)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_directory + '/average_total_rewards.pdf',
                bbox_inches='tight')
    plt.show()
    return


def deep_test(env, output_directory, max_steps=1000, discretize_step=10, agent_type='S-DQN'):
    state_size = env.observation_space.shape[0]

    action_size = discretize_step ** env.num_actions
    dictionary_index_to_action = index_to_action(env.num_actions, discretize_step)
    assert len(dictionary_index_to_action.keys()) == action_size

    agent = StochDQNAgent(state_size, action_size, dictionary_index_to_action, hidden_size=64,
                          learning_rate=0.001, gamma=0.99, epsilon=0,
                          epsilon_decay=0, epsilon_min=0,
                          deterministic=True)

    # Load the trained model
    agent.model.load_state_dict(torch.load(output_directory + '/' + agent_type + '_model.pth'))

    random_actions = np.random.choice(agent.action_size, agent.log2_actions, replace=False)
    random_actions = torch.tensor(random_actions, dtype=torch.int64)

    state = env.reset()[0]
    done = False
    total_reward = 0
    steps = 0
    while not done:
        steps += 1

        action = agent.select_action(state, random_actions)

        action_ = dictionary_index_to_action[action]

        next_state, reward, done, _, _ = env.step(action_)

        observation = state
        next_observation = next_state
        agent.replay_buffer.add((observation, action, reward, next_observation, done))

        state = next_state

        total_reward += reward

        if steps >= max_steps:
            done = True

    env.close()
    return total_reward
