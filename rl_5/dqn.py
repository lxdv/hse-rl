import gym
import gym
import numpy as np
import pandas as pd
gym.logger.set_level(40)

import torch.optim as optim

from tqdm import trange
import matplotlib.pyplot as plt
from dqn_agent_template import DQNAgent
from framebuffer import FrameBuffer
import datetime
from tensorboardX import SummaryWriter

from preprocess import PreprocessAtari
from td_loss_template import compute_td_loss
from replay_buffer import ReplayBuffer


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done: break

        rewards.append(reward)
    return np.mean(rewards)


def play_and_record(agent, env, exp_replay, n_steps=1, render=False):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time
    """
    # initial state
    s = env.framebuffer

    greedy = False
    mean_reward = 0
    # Play the game for n_steps as per instructions above
    for t in range(n_steps):
        qvalues = agent.get_qvalues([s])
        action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]

        n_state, reward, done, info = env.step(action)
        if render:
            env.render()
        mean_reward += reward
        exp_replay.add(s, action, reward, n_state, done=done)
        s = n_state
        if done:
            env.reset()

    return mean_reward


def make_env():
    env = gym.make("BreakoutDeterministic-v4").env
    env = PreprocessAtari(env)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env


def show_output():
    env = make_env()
    env.reset()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape

    for _ in range(50):
        obs, _, _, _ = env.step(env.action_space.sample())

    plt.title("Game image")
    plt.imshow(env.render("rgb_array"))
    plt.show()
    plt.title("Agent observation (4 frames top to bottom)")
    plt.imshow(obs.reshape([-1, state_dim[2]]));
    plt.show()


def dqn():
    log_path = './logs/{:%Y_%m_%d_%H_%M}'.format(datetime.datetime.now())
    writer = SummaryWriter(log_path)
    mean_rw_history, td_loss_history = [], []

    exp_replay = ReplayBuffer(10 ** 5)
    play_and_record(agent, env, exp_replay, n_steps=10000)

    opt = optim.Adam(agent.parameters(), lr=1e-4)

    for i in range(10 ** 5):
        # play
        render = False
        play_and_record(agent, env, exp_replay, 10, render)

        # train
        observs, actions, rewards, observs_n, dones = exp_replay.sample(5)

        loss = compute_td_loss(observs, actions, rewards, observs_n, dones, agent, target_network)
        loss.backward()

        opt.step()
        loss_value = loss.data.cpu().numpy()
        td_loss_history.append(loss_value)

        # adjust agent parameters
        if i % 100 == 0:
            agent.epsilon = max(agent.epsilon * 0.99, 0.01)
            eval_reward = evaluate(make_env(), agent, n_games=3)
            writer.add_scalar('Eval Reward', eval_reward, i)
            print('Eval Reward {}, Step {}'.format(eval_reward, i))
            mean_rw_history.append(eval_reward)

            # Load agent weights into target_network
            target_network.load_state_dict(agent.state_dict())

        if i % 10 == 0:
            # clear_output(True)
            writer.add_scalar('Buffer size', len(exp_replay), i)
            writer.add_scalar('Epsilon', agent.epsilon, i)
            writer.add_scalar('Mean TD loss', loss_value, i)
            assert not np.isnan(td_loss_history[-1])

    assert np.mean(mean_rw_history[-10:]) > 10.
    print("That's good enough for today.")


if __name__ == '__main__':
    # Fill code PreprocessAtari class in preprocess.py, run this file, all checks should be fine.

    # Fill code for DQNAgent class in dqn_agent.py.

    # Fill code for compute_td_loss() in td_loss.py, run this file, all checks should be fine.

    # Now we are ready to run dqn()
    #
    # Note: framebuffer.py should not be modified.
    #       replay_buffer.py should not be modified.

    # Here what you get as input
    # show_output()
    env = make_env()

    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n
    state_dim = observation_shape
    env.reset()
    obs, _, _, _ = env.step(env.action_space.sample())

    agent = DQNAgent(state_dim, n_actions, epsilon=0.5)

    evaluate(env, agent, n_games=1)
    print('testing your code. This may take a minute...')
    exp_replay = ReplayBuffer(20000)
    #
    play_and_record(agent, env, exp_replay, n_steps=1000)
    #
    # if you're using your own experience replay buffer, some of those tests may need correction.
    # just make sure you know what your code does
    assert len(exp_replay) == 1000, "play_and_record should have added exactly 1000 steps, " \
                                     "but instead added %i" % len(exp_replay)
    is_dones = list(zip(*exp_replay._storage))[-1]

    assert 0 < np.mean(
        is_dones) < 0.1, "Please make sure you restart the game whenever it is 'done' and record the is_done correctly into the buffer." \
                         "Got %f is_done rate over %i steps. [If you think it's your tough luck, just re-run the test]" % (
                             np.mean(is_dones), len(exp_replay))

    for _ in range(100):
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(10)
        assert obs_batch.shape == next_obs_batch.shape == (10,) + state_dim
        assert act_batch.shape == (10,), "actions batch should have shape (10,) but is instead %s" % str(
            act_batch.shape)
        assert reward_batch.shape == (10,), "rewards batch should have shape (10,) but is instead %s" % str(
            reward_batch.shape)
        assert is_done_batch.shape == (10,), "is_done batch should have shape (10,) but is instead %s" % str(
            is_done_batch.shape)
        assert [int(i) in (0, 1) for i in is_dones], "is_done should be strictly True or False"
        assert [0 <= a <= n_actions for a in act_batch], "actions should be within [0, n_actions]"

    print("Well done!")

    target_network = DQNAgent(state_dim, n_actions)
    target_network.load_state_dict(agent.state_dict())
    # Let's see the magic of DQN
    dqn()
