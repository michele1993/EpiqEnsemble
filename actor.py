import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as opt
from modelEnsemble import Swish 

class Actor(nn.Module):

    def __init__(self, d_action, d_state, n_units, n_layers, activation, ln_rate, expl_noise, grad_clip, device, normalizer):
        super().__init__()

        self.expl_noise = expl_noise
        self.normalizer = normalizer
        self.grad_clip = grad_clip
        self.device = device
        

        assert n_layers >= 1

        layers = [nn.Linear(d_state, n_units), self.get_activation(activation)]
        for _ in range(1, n_layers):
            layers += [nn.Linear(n_units, n_units), self.get_activation(activation)]
        layers += [nn.Linear(n_units, d_action)]

        [self.init_weights(layer) for layer in layers if isinstance(layer, nn.Linear)]

        self.layers = nn.Sequential(*layers)

        self.to(device)

        self.optimizer = opt.Adam(self.parameters(),ln_rate)

    def init_weights(self, layer):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0)

    def get_activation(self, activation):
        if activation == 'swish':
            return Swish()
        if activation == 'relu':
            return nn.ReLU()
        if activation == 'tanh':
            return nn.Tanh()
        # TODO: I should also initialize depending on the activation
        raise NotImplementedError(f"Unknown activation {activation}")

    # Return deterministic action
    def forward(self, state):
        return torch.tanh(self.layers(state))  # Bound to -1,1

    # Return a exploratory action without gradient
    def expl_action(self, state):

        if self.normalizer is not None:
            state = self.normalizer.normalize_states(state)

        det_a = self(state)

        return  (det_a + torch.randn_like(det_a, device=self.device) * self.expl_noise).clamp(-1,1) # actions bounded [-1,1]

    def update(self, Q):

        loss = -1 * Q.mean()
        # Optimize the actor
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss
