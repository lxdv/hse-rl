import gym
import gym.spaces
import numpy as np


def check_policy_init(initial_policy):
    assert type(initial_policy) in (np.ndarray, np.matrix)
    assert np.allclose(initial_policy, 1. / n_actions)
    assert np.allclose(np.sum(initial_policy, axis=1), 1)
    print('Policy initialization: Ok!')


def check_generate_session_func(generation_func):
    s, a, r = generation_func(policy)
    assert type(s) == type(a) == list
    assert len(s) == len(a)
    assert type(r) in [float, np.float]
    print('Session generation function: Ok!')


def check_update_policy_func(update_func):
    elite_states, elite_actions = ([1, 2, 3, 4, 2, 0, 2, 3, 1], [0, 2, 4, 3, 2, 0, 1, 3, 3])
    new_policy = update_func(elite_states, elite_actions, 5, 6)

    assert np.isfinite(new_policy).all(), 'Your new policy contains NaNs or +-inf. Make sure you do not divide by zero.'
    assert np.all(new_policy >= 0), 'Your new policy should not have negative action probabilities'
    assert np.allclose(new_policy.sum(axis=-1), 1), \
        'Your new policy should be a valid probability distribution over actions'

    reference_answer = np.array([
        [1., 0., 0., 0., 0.],
        [0.5, 0., 0., 0.5, 0.],
        [0., 0.33333333, 0.66666667, 0., 0.],
        [0., 0., 0., 0.5, 0.5]])
    assert np.allclose(new_policy[:4, :5], reference_answer)
    print('Update policy function: Ok!')


def check_select_elites_func(select_elite_func):
    states_batch = [[1, 2, 3], [4, 2, 0, 2], [3, 1]]
    actions_batch = [[0, 2, 4], [3, 2, 0, 1], [3, 3]]
    rewards_batch = [3, 4, 5]

    test_result_0 = select_elite_func(states_batch, actions_batch, rewards_batch, percentile=0)
    test_result_40 = select_elite_func(states_batch, actions_batch, rewards_batch, percentile=30)
    test_result_90 = select_elite_func(states_batch, actions_batch, rewards_batch, percentile=90)
    test_result_100 = select_elite_func(states_batch, actions_batch, rewards_batch, percentile=100)

    assert np.all(test_result_0[0] == [1, 2, 3, 4, 2, 0, 2, 3, 1]) and \
           np.all(test_result_0[1] == [0, 2, 4, 3, 2, 0, 1, 3, 3]), \
        'For percentile 0 you should return all states and actions in chronological order'

    assert np.all(test_result_40[0] == [4, 2, 0, 2, 3, 1]) and \
           np.all(test_result_40[1] == [3, 2, 0, 1, 3, 3]), \
        'For percentile 30 you should only select states/actions from two first'

    assert np.all(test_result_90[0] == [3, 1]) and \
           np.all(test_result_90[1] == [3, 3]), \
        'For percentile 90 you should only select states/actions from one game'

    assert np.all(test_result_100[0] == [3, 1]) and \
           np.all(test_result_100[1] == [3, 3]), \
        'Please make sure you use >=, not >. Also double-check how you compute percentile.'
    print('Select elites function : Ok!')


def generate_session(policy, t_max=10 ** 5):
    """
    Play game until end or for t_max ticks.
    :param policy: an array of shape [n_states,n_actions] with action probabilities
    :returns: list of states, list of actions and sum of rewards
    """
    states, actions = [], []
    total_reward = 0.

    s = env.reset()

    for t in range(t_max):
        # Choose action from policy
        # You can use np.random.choice() func
        # a = ?
        a = np.random.choice(n_actions, p=policy[s])

        # Do action `a` to obtain new_state, reward, is_done,
        new_s, r, is_done, _ = env.step(a)

        # Record state, action and add up reward to states, actions and total_reward accordingly.
        # states
        # actions
        # total_reward

        states.append(s)
        actions.append(a)
        total_reward += r

        # Update s for new cycle iteration
        s = new_s

        if is_done:
            break

    return states, actions, total_reward


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i][t]

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

    Please return elite states and actions in their original order
    [i.e. sorted by session number and timestep within session]

    If you're confused, see examples below. Please don't assume that states are integers (they'll get different later).
    """

    states_batch, actions_batch, rewards_batch = map(np.array, [states_batch, actions_batch, rewards_batch])

    # Compute reward threshold
    reward_threshold = np.percentile(rewards_batch, q=percentile)

    # Compute elite states using reward threshold
    elite_states = states_batch[rewards_batch >= reward_threshold]

    # Compute elite actions using reward threshold
    elite_actions = actions_batch[rewards_batch >= reward_threshold]

    elite_states, elite_actions = map(np.concatenate, [elite_states, elite_actions])

    return elite_states, elite_actions


def update_policy(elite_states, elite_actions, n_states, n_actions):
    """
    Given old policy and a list of elite states/actions from select_elites,
    return new updated policy where each action probability is proportional to

    policy[s_i,a_i] ~ #[occurences of si and ai in elite states/actions]

    Don't forget to normalize policy to get valid probabilities and handle 0/0 case.
    In case you never visited a state, set probabilities for all actions to 1./n_actions

    :param elite_states: 1D list of states from elite sessions
    :param elite_actions: 1D list of actions from elite sessions

    """
    new_policy = np.zeros([n_states, n_actions])

    # Compute updated policy
    for state, action in zip(elite_states, elite_actions):
        new_policy[state][action] += 1

    for i, state in enumerate(new_policy):
        if np.sum(state) > 0:
            new_policy[i] = state / np.sum(state)
        else:
            new_policy[i] = np.ones(n_actions) / n_actions

    return new_policy


def rl_cross_entropy():
    # Useful constants, all should be applied somewhere in your code
    n_sessions = 200  # generate n_sessions for analysis
    percentile = 50  # take this percentage of 'elite' states/actions
    alpha = 0.3  # alpha-blending for policy updates
    total_iterations = 150
    visualize = True
    log = []

    # Create random uniform policy
    policy = np.ones(shape=(n_states, n_actions)) / 6
    check_policy_init(policy)

    if visualize:
        import matplotlib.pyplot as plt
        plt.figure(figsize=[10, 4])

    for i in range(total_iterations):

        # Generate n_sessions for further analysis.
        sessions = [generate_session(policy) for _ in range(n_sessions)]

        states_batch, actions_batch, rewards_batch = zip(*sessions)

        # Select elite states & actions.
        elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile)

        # Update policy using elite_states, elite_actions.
        new_policy = update_policy(elite_states, elite_actions, n_states, n_actions)

        # Alpha blending of old & new policies for stability.
        policy = alpha * new_policy + (1 - alpha) * policy

        # Info for debugging
        mean_reward = np.mean(rewards_batch)
        threshold = np.percentile(rewards_batch, percentile)
        log.append([mean_reward, threshold])

        print('Iteration = %.0f, Mean Reward = %.3f, Threshold = %.3f' % (i, mean_reward, threshold))

        # Visualize training
        if visualize:

            plt.subplot(1, 2, 1)
            plt.plot(list(zip(*log))[0], label='Mean rewards', color='red')
            plt.plot(list(zip(*log))[1], label='Reward thresholds', color='green')

            if i == 0:
                plt.legend()
                plt.grid()

            plt.subplot(1, 2, 2)
            plt.hist(rewards_batch, range=[-990, +10], color='blue', label='Rewards distribution')
            plt.vlines([np.percentile(rewards_batch, percentile)], [0], [100], label='Percentile', color='red')

            plt.legend()
            plt.grid()

            plt.pause(0.1)
            plt.cla()


if __name__ == '__main__':
    # Create environment 'Taxi-v2'
    env = gym.make('Taxi-v2')
    env.reset()
    env.render()

    # Compute number of states for this environment
    n_states = env.observation_space.n
    # Compute number of actions for this environment
    n_actions = env.action_space.n

    print('States number = %i, Actions number = %i' % (n_states, n_actions))

    # Initialize policy - let's say random uniform
    policy = np.ones(shape=(n_states, n_actions)) / n_actions
    check_policy_init(policy)

    # Complete generate session function
    check_generate_session_func(generate_session)

    # Complete select elites function
    check_select_elites_func(select_elites)

    # Complete update policy function
    check_update_policy_func(update_policy)

    # Complete rl_cross_entropy()
    rl_cross_entropy()

    # Close environment when everything is done
    env.close()
    input("Press Enter to continue...")
