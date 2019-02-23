import gym
import gym.spaces  # for warning prevention
import time

from gym.wrappers import TimeLimit
from gym.envs.classic_control import MountainCarEnv
from gym import envs


def brief_overview():
    print('All available environments: ')
    print(envs.registry.all())

    env = gym.make('MountainCar-v0')
    #
    print('Observation space: ', env.observation_space)
    print('Action space: ', env.action_space)
    #
    initial_state = env.reset()

    print('Initial state: ', initial_state)
    #
    print('Taking action 2 (move right):')
    #
    new_state, reward, is_done, _ = env.step(2)
    env.render()
    time.sleep(10)
    #
    print('New state: ', new_state)
    print('Reward after taking an action: ', reward)
    print('Are we done?! ', is_done)


def play_with_car():
    maximum_steps_allowed = 250
    env = TimeLimit(MountainCarEnv(), max_episode_steps=maximum_steps_allowed + 1)
    actions = {'left': 0, 'stop': 1, 'right': 2}

    initial_state = env.reset()
    print('Initial state: ', initial_state)

    for t in range(maximum_steps_allowed):
        # need to modify policy
        if t < 50:
            s, r, done, _ = env.step(actions['left'])
        elif t < 70:
            s, r, done, _ = env.step(actions['right'])
        elif t < 120:
            s, r, done, _ = env.step(actions['left'])
        else:
            s, r, done, _ = env.step(actions['right'])

        print('State {}, Reward {}, Step {}'.format(s, r, t))
        env.render()

        if done:
            if s[0] > 0.47:
                print('Well done!')
            else:
                print('Please, try again.')
            break
    else:
        print('Time is up. Please, try again.')


if __name__ == '__main__':
    print('OpenAI Gym Overview:')

    brief_overview()

    # play_with_car()
