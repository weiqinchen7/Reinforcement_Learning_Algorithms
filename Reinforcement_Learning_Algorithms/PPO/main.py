import gym
import numpy as np
import matplotlib.pyplot as plt
from ppo_torch import Agent


def plot_learning_curve(index, scores, figure_path):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(index, running_avg, 'r')
    plt.grid()
    plt.xlabel('# of episode')
    plt.ylabel('Average score')
    plt.title('Average scores for CartPole (100 smoothing)')
    plt.savefig(figure_path)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    N = 25
    batch_size = 16
    n_epochs = 5
    alpha = 0.0005
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)
    n_games = 300

    figure_path = './cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    index = [i+1 for i in range(len(score_history))]
    plot_learning_curve(index, score_history, figure_path)


