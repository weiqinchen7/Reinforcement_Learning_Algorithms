import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gym
import argparse


class Network(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim):
        super(Network, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(self.n_states, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.n_actions)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = self.out(x)
        return x


class PG(nn.Module):
    def __init__(self, n_states, n_actions, args):
        super(PG, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = args.gamma
        self.lr = args.lr
        self.log_a = []
        self.ep_reward = []
        self.pg_model = Network(self.n_states, self.n_actions, args.hidden_dim)
        self.optimizer = torch.optim.Adam(self.pg_model.parameters(), lr=self.lr)

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)              # expand dimension -> [1, n_states]
        logit = self.pg_model(s)                                  # [1, n_actions]
        prob = F.softmax(logit, 1)                                # activation of logits -> probability
        action = torch.multinomial(prob, 1)                       # Catogorical
        self.log_a.append(torch.log(prob[0][action].squeeze(0)))  # log probability
        return action.item()

    def store(self, r):
        self.ep_reward.append(r)

    def learn(self):
        normalized_ep_reward = np.zeros_like(self.ep_reward)
        temp = 0
        for i in reversed(range(0, len(self.ep_reward))):
           temp = temp * self.gamma + self.ep_reward[i]
           normalized_ep_reward[i] = temp
        epsilon = np.finfo(np.float32).eps.item()
        normalized_ep_reward = (normalized_ep_reward - np.mean(normalized_ep_reward)) / (np.std(normalized_ep_reward) + epsilon)
        normalized_ep_reward = torch.FloatTensor(normalized_ep_reward)

        loss = -torch.sum(torch.cat(self.log_a) * normalized_ep_reward)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_a = []
        self.ep_reward = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    env = gym.make('CartPole-v0')
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = PG(n_states, n_actions, args)
    indice, plot_reward = [], []
    for episode in range(args.n_episodes):
        ep_reward, s = 0, env.reset()
        while True:
            env.render()
            a = agent.choose_action(s)
            s_, r, done, info = env.step(a)
            agent.store(r)
            ep_reward += r
            s = s_
            if done:
                break
        agent.learn()
        print('Episode {} | Reward:{}'.format(episode+1, ep_reward))
        indice.append(episode)
        plot_reward.append(ep_reward)
    plt.plot(indice, plot_reward)
    plt.show()