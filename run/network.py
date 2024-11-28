import torch
import numpy as np
from torch import nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PolicyNetwork(nn.Module):
    """ Policy (Actor) Network """
    def __init__(self, shape_in, action_shape, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.dense1 = layer_init(nn.Linear(shape_in, hidden_size))
        self.dense2 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.dense3_mu = layer_init(nn.Linear(hidden_size, action_shape), std=0.01)
        self.dense3_std = layer_init(nn.Linear(hidden_size, action_shape), std=0.0)

    def forward(self, inputs):
        hid = torch.tanh(self.dense2(torch.tanh(self.dense1(inputs))))
        mu = self.dense3_mu(hid)
        sigma = torch.exp(self.dense3_std(hid))
        return mu, sigma


class ValueNetwork(nn.Module):
    """ Value (Critic) Network """
    def __init__(self, shape_in, hidden_size=64):
        super(ValueNetwork, self).__init__()
        self.dense1 = layer_init(nn.Linear(shape_in, hidden_size))
        self.dense2 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.dense3 = layer_init(nn.Linear(hidden_size, 1), std =1.0)

    def forward(self, inputs):
        hid = torch.tanh(self.dense2(torch.tanh(self.dense1(inputs))))
        return self.dense3(hid)


class Network(nn.Module):
    """ Actot-Critic Network """
    def __init__(self, shape_in, action_shape, hidden_size=64):
        super(Network, self).__init__()
        self.policy = PolicyNetwork(shape_in, action_shape, hidden_size)
        self.value = ValueNetwork(shape_in, hidden_size)

    def forward(self, inputs):
        return self.policy(inputs), self.value(inputs)
