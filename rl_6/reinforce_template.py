import gym
import gym.spaces

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


def test_predict_proba():
    test_states = np.array([env.reset() for _ in range(5)])
    test_probas = predict_proba(test_states)
    assert isinstance(test_probas, np.ndarray), "you must return np array and not %s" % type(test_probas)
    assert tuple(test_probas.shape) == (test_states.shape[0], n_actions), "wrong output shape: %s" % np.shape(test_probas)
    assert np.allclose(np.sum(test_probas, axis = 1), 1), "probabilities do not sum to 1"
    print('Test: predict_proba() function: OK!')


def test_generate_session():
    states, actions, rewards = generate_session()
    assert len(states) == len(actions) == len(rewards), "length must be equal"
    print('Test: generate_session() function: OK!')


def test_get_cumulative_rewards():
    assert len(get_cumulative_rewards(list(range(100)))) == 100
    assert np.allclose(get_cumulative_rewards([0,0,1,0,0,1,0],gamma=0.9),[1.40049, 1.5561, 1.729, 0.81, 0.9, 1.0, 0.0])
    assert np.allclose(get_cumulative_rewards([0,0,1,-2,3,-4,0],gamma=0.5), [0.0625, 0.125, 0.25, -1.5, 1.0, -4.0, 0.0])
    assert np.allclose(get_cumulative_rewards([0,0,1,2,3,4,0],gamma=0), [0, 0, 1, 2, 3, 4, 0])
    print('Test: get_cumulative_rewards() function: OK!')


# < YOUR CODE HERE >
# Build a simple neural network that predicts policy logits.
# Keep it simple: CartPole isn't worth deep architectures.
class ReinforceAgent(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(ReinforceAgent, self).__init__()
        self.fc1 = nn.Linear(state_dim[0], 42)
        self.fc2 = nn.Linear(42, 102)
        self.fc3 = nn.Linear(102, 42)
        self.fc4 = nn.Linear(42, n_actions)
        pass

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# < YOUR CODE HERE >
def predict_proba(states):
    """
    Predict action probabilities given states.
    :param states: numpy array of shape [batch, state_shape]
    :returns: numpy array of shape [batch, n_actions]
    """
    # convert states, compute logits, use softmax to get probability
    predicted = agent(torch.Tensor(states))
    probs = F.softmax(predicted).data.numpy()
    return probs


# < YOUR CODE HERE >
def generate_session(t_max=1000):
    """
    play a full session with REINFORCE agent and train at the session end.
    returns sequences of states, actions andrewards
    """
    # arrays to record session
    states, actions, rewards = [], [], []
    s = env.reset()

    for t in range(t_max):

        # action probabilities array aka pi(a|s)
        action_probas = predict_proba([s])[0]
        a = np.random.choice(n_actions, p=action_probas)
        new_s, r, done, info = env.step(a)

        # record session history to train later
        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = new_s
        if done:
            break

    return states, actions, rewards


# < YOUR CODE HERE >
def get_cumulative_rewards(rewards,  # rewards at each step
                           gamma=0.99  # discount for reward
                              ):
    """
    take a list of immediate rewards r(s,a) for the whole session
    compute cumulative returns (a.k.a. G(s,a) in Sutton '16)
    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    The simple way to compute cumulative rewards is to iterate from last to first time tick
    and compute G_t = r_t + gamma*G_{t+1} recurrently

    You must return an array/list of cumulative rewards with as many elements as in the initial rewards.
    """
    cumulative_rewards = []
    prev = 0

    for r in reversed(rewards):
        prev = r + gamma * prev
        cumulative_rewards.append(prev)
    cumulative_rewards.reverse()
    return cumulative_rewards

# Helper function
def to_one_hot(y, n_dims=None):
    """ Take an integer vector (tensor of variable) and convert it to 1-hot matrix. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


# < YOUR CODE HERE >
def train_on_session(optimizer, states, actions, rewards, gamma=0.99):
    """
    Takes a sequence of states, actions and rewards produced by generate_session.
    Updates agent's weights by following the policy gradient above.
    Please use Adam optimizer with default parameters.
    """

    # cast everything into a variable
    states = Variable(torch.FloatTensor(states))
    actions = Variable(torch.IntTensor(actions))
    cumulative_returns = np.array(get_cumulative_rewards(rewards, gamma))
    cumulative_returns = Variable(torch.FloatTensor(cumulative_returns))

    # predict logits, probas and log-probas using an agent.
    logits = agent(states)
    probas = F.softmax(logits, dim=1)
    logprobas = F.log_softmax(logits, dim=1)

    assert all(isinstance(v, Variable) for v in [logits, probas, logprobas]), \
        "please use compute using torch tensors and don't use predict_proba function"

    # select log-probabilities for chosen actions, log pi(a_i|s_i)
    logprobas_for_actions = torch.sum(logprobas * to_one_hot(actions), dim=1)

    # REINFORCE objective function
    J_hat = torch.mean(logprobas_for_actions * cumulative_returns)

    agent.zero_grad()
    loss = -J_hat
    loss.backward()
    optimizer.step()

    # technical: return session rewards to print them later
    return np.sum(rewards)


if __name__ == '__main__':
    env = gym.make("CartPole-v0").env

    env.reset()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape

    env.render("rgb_array")
    env.close()

    # 1. Complete ReinforceAgent class
    # 2. Complete predict_proba()
    # 3. Complete generate_session()
    # 4. Complete get_cumulative_rewards()
    # 5. Complete train_on_sessions()

    # Create agent
    agent = ReinforceAgent(state_dim, n_actions)
    test_predict_proba()
    test_generate_session()

    test_get_cumulative_rewards()

    # call train_on_sessions()
    for i in range(100):
        optimizer = optim.Adam(agent.parameters())
        rewards = []
        for _ in range(100):
            session = generate_session()
            rewards.append(train_on_session(optimizer, *session))

        print("Iteration: %i, Mean reward:%.3f" % (i, np.mean(rewards)))

        if np.mean(rewards) > 500:
            print("You Win!")  # but you can train even further
            break