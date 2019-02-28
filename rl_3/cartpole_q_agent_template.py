import gym
import gym.spaces
import math
from qlearning_template import QLearningAgent
import numpy as np
import time


def discretize_range(lower_bound, upper_bound, num_bins):
    return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]


def discretize_value(value, bins):
    return np.digitize(x=value, bins=bins)


def build_state(observation):
    return sum(discretize_value(feature, state_bins[i]) * ((max_bins + 1) ** i) for i, feature in enumerate(observation))


def play_and_train(env, agent, visualize=False, t_max=10 ** 4):
    """This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total reward"""
    total_reward = 0.0
    s = env.reset()

    for t in range(t_max):
        d_s = build_state(s)

        a = agent.get_action(d_s)

        next_s, r, done, _ = env.step(a)
        if visualize:
            env.render()
            time.sleep(0.05)
        d_next_s = build_state(next_s)
        agent.update(d_s, a, d_next_s, r)
        s = next_s
        total_reward += r
        if done:
            break

    return total_reward


if __name__ == '__main__':
    env = gym.make("CartPole-v0").env
    env.reset()
    n_actions = env.action_space.n

    print(env.observation_space.high)
    print(env.observation_space.low)
    print('CartPole state: %s' % (env.reset()))

    agent = QLearningAgent(0.3, 0.5, 1.0, lambda s: range(n_actions))

    # (x, x', theta, theta')
    state_bins = [  # Cart position.
        discretize_range(-2.4, 2.4, 2),
        # Cart velocity.
        discretize_range(-3.0, 3.0, 2),
        # Pole angle.
        discretize_range(-0.5, 0.5, 7),
        # Tip velocity.
        discretize_range(-2.0, 2.0, 7)
    ]
    max_bins = max(len(bin) for bin in state_bins)

    rewards = []
    for i in range(2000):
        rewards.append(play_and_train(env, agent))
        agent.epsilon *= 0.999

        if i % 10 == 0:
            print('Iteration {}, Average reward {:.2f}, Epsilon {:.3f}'.format(i, np.mean(rewards), agent.epsilon))

    print('Reward of Test agent = %.3f' % play_and_train(env, agent, visualize=True))
