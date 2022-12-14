# taken from OpenAI baselines.

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.core import Wrapper
from gymnasium.core import ObservationWrapper

########### from atari_wrappers.py #################

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, trunc,info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, trunc,info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class ScaleRewardEnv(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale

    def reward(self, reward):
        return reward * 0.01


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs, None

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, trunc,info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _,_ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, None


# in torch imgs have shape [c, h, w] instead of common [h, w, c]
class AntiTorchWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

        self.img_size = [env.observation_space.shape[i]
                         for i in [1, 2, 0]
                         ]
        self.observation_space = gym.spaces.Box(0.0, 1.0, self.img_size)

    def observation(self, img):
        """what happens to each observation"""
        img = img.transpose(1, 2, 0)
        return img


########### from framebuffer.py #################

class FrameBuffer(Wrapper):
    def __init__(self, env, n_frames=4):
        """A gym wrapper that reshapes, crops and scales image into the desired shapes"""
        super(FrameBuffer, self).__init__(env)
        n_channels, height, width = env.observation_space.shape
        obs_shape = [n_channels * n_frames, height, width]
        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.framebuffer = np.zeros(obs_shape, 'uint8')

    def reset(self):
        """resets breakout, returns initial frames"""
        self.framebuffer = None

        state,info = self.env.reset()
        self.update_buffer(state)
        return self.framebuffer

    def step(self, action):
        """plays breakout for 1 step, returns frame buffer"""
        new_img, reward, done, trunc,info = self.env.step(action)
        """
         on reset(1,42,42)
         on step (42,42)
         why? 
        """ 
       # print("Frame buffer : {}".format(new_img.shape))
       # self.update_buffer(np.expand_dims(new_img,0))
        self.update_buffer(new_img)

        return self.framebuffer, reward, done, trunc,info

    def update_buffer(self, img):
        #print("update buffer : {} {}".format(type(img),img))
        if self.framebuffer is None:
            self.framebuffer = np.repeat(img, 4, axis=0)

        self.framebuffer = np.append(self.framebuffer[1:, :, :], img, axis=0)

        # mod
        #self.framebuffer = np.append(self.framebuffer[1:], img, axis=0)


######### from KungFuMasterDeterministic.py #######



class PreprocessAtariObs(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and grayscales it."""
        ObservationWrapper.__init__(self, env)
        self.env = env

        self.img_size = (1, 42, 42)
        self.observation_space = Box(0.0, 1.0, self.img_size)

    def observation(self, img):
        """what happens to each observation"""

        #print("obs : {} {}".format(type(img),img[0].shape))

        ## TODO : Delete this. 
        if type(img) is tuple :
            img = img[0]

        self.env.raw = np.copy(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img[60:-30, 5:]
        img = cv2.resize(img, (42, 42), cv2.INTER_NEAREST)

        return img.reshape(-1, 42, 42)


def PrimaryAtariWrap(env, clip_rewards=True, scale=100, disp=False,real=False):
    # This wrapper holds the same action for <skip> frames and outputs
    # the maximal pixel value of 2 last frames (to handle blinking
    # in some envs)
    env = MaxAndSkipEnv(env, skip=1)

    if not real : 
    # This wrapper sends done=True when each life is lost
    # (not all the 5 lives that are givern by the game rules).
    # It should make easier for the agent to understand that losing is bad.
        env = EpisodicLifeEnv(env)

    # This wrapper laucnhes the ball when an episode starts.
    # Without it the agent has to learn this action, too.
    # Actually it can but learning would take longer.
    # env = atari_wrappers.FireResetEnv(env)

    # This wrapper transforms rewards to {-1, 0, 1} according to their sign
    # if clip_rewards:
    #     env = atari_wrappers.ScaleRewardEnv(env, scale=100)

    if not disp :
        env = PreprocessAtariObs(env)
    return env


def make_env(ENV_NAME,clip_rewards=True, seed=None,disp=False,real=False):
    #env = gym.make(ENV_NAME,obs_type="grayscale")  # create raw env
    #env = gym.make("GymV26Environment-v0",env_id=ENV_NAME)

    # https://github.com/Farama-Foundation/Gymnasium/issues/152
    # pip install shimmy[atari]


    # https://github.com/Farama-Foundation/Gymnasium/issues/77
    #env = gym.make(ENV_NAME,apply_api_compatibility=True)
    env = gym.make(ENV_NAME)
    if seed is not None:
        env.seed(seed)
    env = PrimaryAtariWrap(env, clip_rewards=clip_rewards,disp=disp,real=real)
    if not disp :
        env = FrameBuffer(env, n_frames=4)
    return env