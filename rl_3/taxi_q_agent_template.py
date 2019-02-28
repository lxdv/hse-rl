import gym.spaces
import matplotlib.pyplot as plt
import numpy as np
from qlearning_template import QLearningAgent


def play_and_train(env, agent, t_max=10 ** 4):
    """ This function should
    - run a full game (for t_max steps), actions given by agent
    - train agent whenever possible
    - return total reward
    """
    total_reward = 0.0
    state = env.reset()
    for step in range(t_max):
        a = agent.get_action(state)
        new_state, reward, done, _ = env.step(a)
        agent.update(state, a, new_state, reward)
        total_reward += reward
        state = new_state
        if done:
            break

    return total_reward


if __name__ == '__main__':
    max_iterations = 1000
    visualize = True
    # Create Taxi-v2 env
    env = gym.make("Taxi-v2")
    env.reset()
    env.render()

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    print('States number = %i, Actions number = %i' % (n_states, n_actions))

    # create q learning agent with
    # alpha=0.5
    # get_legal_actions = lambda s: range(n_actions)
    # epsilon=0.1
    # discount=0.99

    alpha = 0.5
    get_legal_actions = lambda s: range(n_actions)
    epsilon = 0.1
    discount = 0.99

    agent = QLearningAgent(alpha, epsilon, discount, get_legal_actions)

    plt.figure(figsize=[10, 4])
    rewards = []

    # Training loop
    for i in range(max_iterations):
        # Play & train game
        # Update rewards
        # rewards

        # Decay agent epsilon
        # agent.epsilon = ?

        initial_state = env.reset()
        rewards.append(play_and_train(env, agent))
        agent.epsilon = agent.epsilon * discount

        if i % 100 == 0:
            print('Iteration {}, Average reward {:.2f}, Epsilon {:.3f}'.format(i, np.mean(rewards), agent.epsilon))

        if visualize:
            plt.subplot(1, 2, 1)
            plt.plot(rewards, color='r')
            plt.xlabel('Iterations')
            plt.ylabel('Total Reward')

            plt.subplot(1, 2, 2)
            plt.hist(rewards, bins=20, range=[-700, +20], color='blue', label='Rewards distribution')
            plt.xlabel('Reward')
            plt.ylabel('p(Reward)')
            plt.draw()
            plt.pause(0.01)
            plt.cla()
