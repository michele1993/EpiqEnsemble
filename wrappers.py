import numpy as np
import torch

import gym
from packaging import version
from utils import to_torch


class BoundedActionsEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.unwrapped.action_space.shape)

    def step(self, action):
        action = np.clip(action, -1., 1.)
        lb, ub = self.unwrapped.action_space.low, self.unwrapped.action_space.high
        scaled_action = lb + (action + 1.0) * 0.5 * (ub - lb)
        observation, reward, done, info = self.env.step(scaled_action)
        return observation, reward, done, info

class IsDoneEnv(gym.Wrapper):
    """
        Wrapper correcting the "done" computation as the done computed by other wrappers is computed based on
        raw states - when the NoisyEnv wrapper is applied on top of such states the noisyfied state done value
        can differ from the non-noisyfied state done value
        NOTE: For the fix to work this wrapper need to be the last wrapper applied to the Env  # TODO: It is not. Is that comment correct?
    """

    def __init__(self, env):
        assert hasattr(env.unwrapped, 'is_done'), f"{env} has no is_done method to compute done caused by termination conditions"
        super().__init__(env)

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        with torch.no_grad():  # Just to be sure...
            to_torch(state)
            done_condition = bool(self.env.unwrapped.is_done(t_state).item())  # done do to termination conditions
        done_timelimit = done if done else False
        done = done_condition or done_timelimit

        return state, reward, done, info


class MuJoCoCloseFixWrapper(gym.Wrapper):
    def close(self):
        success = False
        try:
            # A fix for MuJoCo (https://github.com/openai/gym/issues/1000)
            from gym.envs.mujoco import mujoco_env
            if isinstance(self.env.unwrapped, mujoco_env.MujocoEnv):
                if hasattr(self, 'viewer') and self.viewer is not None and hasattr(self.viewer, 'window'):
                    import glfw
                    if isinstance(self.viewer.window, glfw._GLFWwindow):
                        glfw.destroy_window(self.viewer.window)
                        self.viewer = None
                        success = True
        finally:
            pass

        # Either not a MuJoCo env or some other problem. Fall back to the standard behaviour
        if not success:
            super().close()
