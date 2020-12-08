import gym
from gym import spaces
import numpy as np
from scipy import ndimage

import os
from PIL import Image
import math

from collections import defaultdict, deque

class Support_v0(gym.Env):
    def __init__(self, dataset_dir, board_size):
        super().__init__()

        self.board_size = board_size
        self.dataset_dir = dataset_dir
        _, _, self.filenames = next(os.walk(self.dataset_dir))

        # feature shape
        # 1) empty positions lower than the action row
        # 2) the upper row
        # 3) the action row
        # 4) legal action
        self.obs_shape = (4, self.board_size, self.board_size)

        self.action_space = spaces.Discrete(self.board_size)
        self.observation_space = spaces.Box(0.0, 1.0, self.obs_shape, dtype=np.float32)

    def is_valid_action(self, action):
        if self.model[self.action_row, action] or self.support[self.action_row, action]:
            return False
        else:
            return True

    def step(self, action):
        if not self.is_valid_action(action):
            return self.obs(), -9999.0, False, {}

        self.support[self.action_row, action] = True

        # if self.update_action_row():
        #     rwd = prev_unsupported-1
        # else:
        #     delta = prev_unsupported - self.unsupported
        #     rwd = -100 if delta == 0 else delta-1

        is_straight = self.support[self.action_row-1, action]
        rwd = -0.05
        rwd = rwd+0.05 if is_straight else rwd
        self.update_action_row()

        # if self.update_action_row():
        #     rwd = 0 if prev_unsupported == 1 else 0.01
        # else:
        #     delta = prev_unsupported - self.unsupported
        #     if delta == 0:
        #         rwd = -0.2
        #     elif delta == 1:
        #         rwd = 0
        #     else:
        #         rwd = 0.01

        if self.action_row == self.board_size:
            return self.obs(True), rwd, True, {}
        else:
            return self.obs(), rwd, False, {}

    def reset(self):
        while True:
            filename = np.random.choice(self.filenames, 1)
            img = Image.open(self.dataset_dir + filename[0]).convert("L")
            self.model = np.array(img, dtype=np.bool)
            self.support = np.zeros_like(self.model)

            self.action_row = 1
            self.update_action_row()
            if self.action_row != self.board_size:
                break

        return self.obs()

    def count_unsupported(self, row):
        upper = np.logical_or(self.model[row, :], self.support[row, :])
        lower = np.logical_or(self.model[row + 1, :], self.support[row + 1, :])

        dilated = ndimage.binary_dilation(lower)

        res = np.logical_and(upper, np.logical_not(dilated))
        unsupported = np.sum(res)
        return unsupported

    def update_action_row(self):
        action_row_change = False
        while self.action_row < self.board_size:
            self.unsupported = self.count_unsupported(self.action_row - 1)
            if self.unsupported == 0:
                self.action_row += 1
                action_row_change = True
            else:
                break

        return action_row_change

    def empty_position_feature(self):
        filled = np.logical_or(self.model, self.support)

        # empty feature
        feature = np.logical_not(filled)

        # no need to consider rows upper than the action row
        feature[0 : self.action_row, :] = False
        return feature

    def row_feature(self, row):
        filled = np.logical_or(self.model[row, :], self.support[row, :])

        # tile the row along the axis 0
        feature = np.tile(filled, (self.board_size, 1))
        return feature

    def upper_row_feature(self):
        return self.row_feature(self.action_row - 1)
    
    def upper_support_row_feature(self):
        row = self.action_row - 1
        filled = self.support[row, :]

        # tile the row along the axis 0
        feature = np.tile(filled, (self.board_size, 1))
        return feature

    def action_row_feature(self):
        return self.row_feature(self.action_row)

    def legal_action_feature(self):
        filled = np.logical_or(
            self.model[self.action_row, :], self.support[self.action_row, :]
        )
        empty = np.logical_not(filled)
        feature = np.zeros_like(self.model)
        feature[self.action_row, np.nonzero(empty)[0]] = True
        return feature

    def obs(self, terminal=False):
        ret = np.zeros(self.obs_shape, dtype=np.bool)
        # ret[0] = self.model
        # ret[1] = self.support
        if not terminal:
            ret[0] = self.empty_position_feature()
            ret[1] = self.upper_row_feature()
            ret[2] = self.action_row_feature()
            ret[3] = self.legal_action_feature()
        return ret.astype(np.float32)

if __name__ == "__main__":

    env = Support_v0(dataset_dir = "../size30/train/", board_size = 30)
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)