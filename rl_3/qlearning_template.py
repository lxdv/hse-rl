from collections import defaultdict
import numpy as np
import random


class QLearningAgent:
    """
      Q-Learning Agent
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate aka gamma)
      Functions you should use
        - self.get_legal_actions(state)
          which returns legal actions for a state
        - self.get_q_value(state,action)
          which returns Q(state,action)
        - self.set_q_value(state,action,value)
          which sets Q(state,action) := value

      !!!Important!!!
      NOTE: please avoid using self._qValues directly to make code cleaner
    """

    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """We initialize agent and Q-values here."""
        self.get_legal_actions = get_legal_actions
        self._qValues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = discount

    def get_q_value(self, state, action):
        """
          Returns Q(state,action)
        """
        return self._qValues[state][action]

    def set_q_value(self, state, action, value):
        """
          Sets the Qvalue for [state,action] to the given value
        """
        self._qValues[state][action] = value

    ################ YOUR CODE is here! ##################

    def get_state_value(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.
        """

        possible_actions = self.get_legal_actions(state)
        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        q_values = []
        for action in possible_actions:
            q_values.append(self.get_q_value(state, action))

        return np.max(q_values)


    def get_best_policy_action(self, state):
        """
          Compute the best action to take in a state according current policy

        """
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None

        q_values = []
        for actions in possible_actions:
            q_values.append(self.get_q_value(state, actions))

        return np.argmax(q_values)

    def get_action(self, state):
        """
          Compute the action to take in the current state, including exploration.

          With probability self.epsilon, we should take a random action.
          otherwise - the best policy action (self.getPolicy).
          HINT: You might want to use random.random() or random.choice(list)
        """
        # Pick Action
        possible_actions = self.get_legal_actions(state)
        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        r = random.random()
        if r <= self.epsilon:
            return random.choice(possible_actions)

        return self.get_best_policy_action(state)

    def update(self, state, action, next_state, reward):
        """
          You should do your Q-Value update here
          NOTE: You should never call this function inside class,
          it will be called later
        """
        delayed_reward = self.gamma * self.get_state_value(next_state)
        updated_q = self.alpha * (reward + delayed_reward) + (1 - self.alpha) * self.get_q_value(state, action)
        self.set_q_value(state, action, updated_q)
