import numpy as np
import gym
import gridworld

class QLearning(object):
    def __init__(self, n_status, n_act, lr, gamma, epsilon):
        self.n_status = n_status
        self.n_act = n_act
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q_table = np.zeros((n_status, n_act))  # initialization

    # exploitation
    def action_exploit(self, state):
        tmp = self.Q_table[state, :]
        action = np.random.choice(np.flatnonzero(tmp == tmp.max()))  # randomly choose the maximizer, instead of just choosing the first one
        return action

    # exploration & exploitation
    def action_explore(self, state):
        if np.random.uniform(0, 1) < self.epsilon:  # exploration
            action = np.random.choice(self.n_act)
        else:  # exploitation
            action = self.action_exploit(state)
        return action

    # learn and update Q_table
    def learn(self, state, action, reward, state_, done):
        current_Q = self.Q_table[state, action]
        target_Q = reward + (1 - int(done)) * self.gamma * self.Q_table[state_, :].max()
        self.Q_table[state, action] += self.lr * (target_Q - current_Q)

# train one episode
def train_episode(env, agent, is_render):
    ep_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = agent.action_explore(state)  # action needs exploration
        state_, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, state_, done)
        state = state_
        ep_reward += reward
        if is_render:
            env.render()  # render every 'is_render' step
    return ep_reward

# test one episode
def test_episode(env, agent):
    ep_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = agent.action_exploit(state)  # action only does exploitation during testing
        state_, reward, done, _ = env.step(action)
        ep_reward += reward
        state = state_
        env.render()  # render at every step
    return ep_reward

# training & testing
def train_test(env, episodes, lr, gamma, epsilon):
    agent = QLearning(
        n_status = env.observation_space.n,
        n_act = env.action_space.n,
        lr = lr,
        gamma = gamma,
        epsilon = epsilon)

    # training
    is_render = False
    for i in range(episodes):
        ep_reward = train_episode(env, agent,is_render)
        print('Episode %d: ep_reward = %.2f' % (i, ep_reward))
        if i % 50 == 0:
            is_render = True
        else:
            is_render = False

    # testing
    test_reward = test_episode(env, agent)
    print('Final testing reward = %.2f' % (test_reward))


if __name__ == '__main__':
    # hyper_parameters
    episodes = 600
    lr = 0.1
    gamma = 0.95
    epsilon = 0.05
    env = gym.make("CliffWalking-v0")
    env = gridworld.CliffWalkingWapper(env)  # wapper by PaddlePaddle
    train_test(env, episodes, lr, gamma, epsilon)