import gym
from gym.utils.play import play

if __name__ == "__main__":
    env = gym.make("BreakoutDeterministic-v0")  # create raw env
    play(env, fps=8, zoom=3)