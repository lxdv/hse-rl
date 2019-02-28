import gym
from gym.core import ObservationWrapper
from gym.spaces import Box
import cv2
import numpy as np
import matplotlib.pyplot as plt


class PreprocessAtari(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.img_size = (1, 64, 64)
        self.observation_space = Box(0.0, 1.0, self.img_size)

    def _observation(self, img):
        """what happens to each observation"""

        # Here's what you need to do:
        #  * crop image, remove irrelevant parts
        #  * resize image to self.img_size
        #     (use imresize imported above or any library you want,
        #      e.g. opencv, skimage, PIL, keras)
        #  * cast image to grayscale
        #  * convert image pixels to (0,1) range, float32 type

        #img = img[16:-16, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.img_size[1], self.img_size[1]))
        img = img / 255

        return np.expand_dims(img.astype(np.float32), axis=0)


if __name__ == '__main__':
    # spawn game instance for tests
    env = gym.make("BreakoutDeterministic-v0")  # create raw env
    env = PreprocessAtari(env)

    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n

    env.reset()
    obs, _, _, _ = env.step(env.action_space.sample())

    # test observation
    assert obs.ndim == 3, "observation must be [batch, time, channels] even if there's just one channel"
    assert obs.shape == observation_shape
    assert obs.dtype == 'float32'
    assert len(np.unique(obs)) > 2, "your image must not be binary"
    assert 0 <= np.min(obs) and np.max(obs) <= 1, "convert image pixels to (0,1) range"

    print("Formal tests seem fine. Here's an example of what you'll get.")

    plt.title("what your network gonna see")
    plt.imshow(obs[0, :, :], interpolation='none', cmap='gray');
    plt.show()
