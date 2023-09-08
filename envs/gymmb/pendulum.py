import torch
import numpy as np
from gym import Env
from gym.envs.classic_control import PendulumEnv

class GYMMB_Pendulum(PendulumEnv):

    def __init__(self):
        super().__init__()

    @staticmethod
    def is_done(states):
        es = states.shape[0]
        bs = states.shape[1]
        return torch.zeros(size=(es,bs,1), dtype=torch.bool, device=states.device)  # Always False


    @staticmethod
    def normalize_angle(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def reward(self, states, actions, next_states):
        max_torque = 2.

        torque = torch.clamp(actions, min=-max_torque, max=max_torque)[0]

        costheta, sintheta, thetadot = states[:, 0], states[:, 1], states[:, 2]
        theta = self.normalize_angle(torch.atan2(sintheta, costheta))
        cost = theta.pow(2) + .1 * thetadot.pow(2) + .001 * torque.pow(2)
        return -cost
