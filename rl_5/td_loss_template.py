import torch
from torch.autograd import Variable
import numpy as np
import gym
from dqn_agent_template import DQNAgent
from preprocess import PreprocessAtari
from replay_buffer import ReplayBuffer
from framebuffer import FrameBuffer


def compute_td_loss(states, actions, rewards, next_states, is_done, agent, target_network, gamma=0.99, check_shapes=False):
    """ Compute td loss using torch operations only. Use the formula above. """
    states = Variable(torch.FloatTensor(states))  # shape: [batch_size, c, h, w]
    actions = Variable(torch.LongTensor(actions))  # shape: [batch_size]
    rewards = Variable(torch.FloatTensor(rewards))  # shape: [batch_size]
    next_states = Variable(torch.FloatTensor(next_states))  # shape: [batch_size, c, h, w]
    is_done = Variable(torch.FloatTensor(is_done.astype('float32')))  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_qvalues = agent(states)

    # compute q-values for all actions in next states
    predicted_next_qvalues = target_network(next_states)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]

    # compute V*(next_states) using predicted next q-values
    next_state_values = torch.max(predicted_next_qvalues, dim=1)[0]

    next_state_values = next_state_values * is_not_done

    assert next_state_values.dim() == 1 and next_state_values.shape[0] == states.shape[
        0], "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    target_qvalues_for_actions = rewards + gamma * next_state_values

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)

    if check_shapes:
        assert predicted_next_qvalues.data.dim() == 2, "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim() == 1, "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim() == 1, "there's something wrong with target q-values, they must be a vector"

    return loss


if __name__ == '__main__':
    env = gym.make("BreakoutDeterministic-v0")  # create raw env
    env = PreprocessAtari(env)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')

    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n
    state_dim = observation_shape
    env.reset()
    obs, _, _, _ = env.step(env.action_space.sample())
    agent = DQNAgent(state_dim, n_actions, epsilon=0.5)
    target_network = DQNAgent(state_dim, n_actions)

    exp_replay = ReplayBuffer(10)
    for _ in range(30):
        exp_replay.add(env.reset(), env.action_space.sample(), 1.0, env.reset(), done=False)

    target_network.load_state_dict(agent.state_dict())
    # sanity checks
    obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(10)

    loss = compute_td_loss(obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch, gamma=0.99,
                           check_shapes=True)
    loss.backward()

    assert np.any(next(agent.parameters()).grad.data.numpy() != 0), "loss must be differentiable w.r.t. network weights"
    print("TD Loss OK")
