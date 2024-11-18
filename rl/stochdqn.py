import random
from math import inf

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch import mean

print('cuda', torch.cuda.is_available())


class SQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, use_cuda=True):
        super(SQNetwork, self).__init__()
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fcq = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fcq(x)
        return x


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, experience):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.vstack(states), actions, rewards, np.vstack(next_states), dones


# Define the StochDQN agent
class StochDQNAgent:
    def __init__(self, state_size, action_size, dictionary_index_to_action, hidden_size=64, learning_rate=0.001,
                 gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, deterministic=True, use_cuda=True, double=True):

        print('Q Network with action and state input.')

        self.state_size = state_size
        self.action_size = action_size

        self.input_size = self.state_size + len(dictionary_index_to_action[0])

        self.log2_actions = round(np.log2(action_size))
        print('log2(A) ', round(np.log2(action_size)))

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        self.dictionary_index_to_action = dictionary_index_to_action
        self.batch_size = 2 * self.log2_actions
        self.buffer_size = 2 * self.batch_size
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.deterministic = deterministic

        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print('device ', self.device)

        self.model = SQNetwork(self.input_size, hidden_size, use_cuda=use_cuda).to(self.device)
        self.target_model = SQNetwork(self.input_size, hidden_size, use_cuda=use_cuda).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.loss_fn = nn.MSELoss()
        self.target_model.eval()

        self.double = double

    def stochastic_maximum(self, state_tensor, action_set, target=False):
        best_action = [action_set[0]] * state_tensor.shape[0]
        q_max = torch.tensor([-float(inf)] * state_tensor.shape[0], dtype=torch.float32,
                             device=self.device)

        action_set = set(action_set)

        for action in action_set:
            action_tensor = torch.tensor(self.dictionary_index_to_action[action], dtype=torch.float32,
                                         device=self.device)

            action_tensor = action_tensor.repeat(state_tensor.shape[0], 1)

            x = torch.cat((state_tensor, action_tensor), dim=1)

            if target:
                q_value = self.target_model(x)
            else:
                q_value = self.model(x)

            for i in range(state_tensor.shape[0]):
                if q_value[i] > q_max[i]:
                    q_max[i] = q_value[i].item()
                    best_action[i] = action

        return q_max, best_action

    def select_action(self, state, random_actions):
        if np.random.rand() <= self.epsilon or len(self.replay_buffer.buffer) < self.batch_size:
            random_index = random.randrange(self.action_size)
            return random_index
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

                if len(state_tensor.shape) == 1:
                    state_tensor = state_tensor.unsqueeze(0)

                if self.deterministic:
                    _, best_action = self.stochastic_maximum(state_tensor, range(self.action_size))

                else:
                    _, batch_actions, _, _, _ = self.replay_buffer.sample(self.batch_size)
                    actions_set = np.concatenate((batch_actions, random_actions), axis=0)
                    _, best_action = self.stochastic_maximum(state_tensor, actions_set)

                return best_action[0]

    def train(self, state, action, reward, next_state, done, random_actions):

        self.replay_buffer.add((state, action, reward, next_state, done))

        if len(self.replay_buffer.buffer) >= self.batch_size:
            max_reps = 1
            repetition = 0
            while repetition < max_reps:
                repetition += 1

                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.replay_buffer.sample(
                    self.batch_size)

                state_tensor = torch.tensor(batch_states, dtype=torch.float32, device=self.device)
                next_state_tensor = torch.tensor(batch_next_states, dtype=torch.float32, device=self.device)
                batch_dones = torch.tensor([1 if done else 0 for done in batch_dones], dtype=torch.float32,
                                           device=self.device)
                batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device)
                batch_actions_ = torch.tensor([self.dictionary_index_to_action[action] for action in batch_actions])

                x = np.concatenate((state_tensor, batch_actions_), axis=1)
                x = torch.tensor(x, dtype=torch.float32, device=self.device)

                if self.deterministic:
                    q_max, _ = self.stochastic_maximum(next_state_tensor, range(self.action_size), target=self.double)
                else:
                    actions_set = np.concatenate((batch_actions, random_actions), axis=0)
                    q_max, _ = self.stochastic_maximum(next_state_tensor, actions_set, target=self.double)

                q_values = batch_rewards_tensor + self.gamma * q_max * (1 - batch_dones)

                self.optimizer.zero_grad()

                loss = self.loss_fn(q_values.unsqueeze(1), self.model(x))
                # print(loss)
                loss.backward()
                self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
