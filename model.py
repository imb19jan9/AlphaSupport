import gym

import torch as th
import torch.nn as nn

import numpy as np

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


class ResidualBlock(nn.Module):
    def __init__(self, n_channel):
        super(ResidualBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(n_channel, n_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(),
            nn.Conv2d(n_channel, n_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_channel),
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        out = x + self.net(x)
        return self.activation(out)


class ResFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, n_channel, n_block):
        super(ResFeatureExtractor, self).__init__(observation_space, 1)

        self.conv_in = nn.Sequential(
            nn.Conv2d(
                observation_space.shape[0],
                n_channel,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(),
        )

        layers = []
        for _ in range(n_block):
            layers.append(ResidualBlock(n_channel))
        self.res_block = nn.Sequential(*layers)

        self.conv_out = nn.Sequential(
            nn.Conv2d(n_channel, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.conv_in(observations)
        x = self.res_block(x)
        return self.conv_out(x)


class Res_ValueHead(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(Res_ValueHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class Res_PolicyHead(nn.Module):
    def __init__(self, feature_dim, n_actions):
        super(Res_PolicyHead, self).__init__()

        self.net = nn.Sequential(nn.Linear(feature_dim, n_actions))

    def forward(self, x):
        return self.net(x)