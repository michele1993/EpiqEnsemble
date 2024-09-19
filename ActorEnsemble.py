import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from modelEnsemble import EnsembleLayer 
import torch.optim as opt

class ActorEnsemble(nn.Module):

    def __init__(self, d_action, d_state, n_units, n_layers, ensemble_size, activation,ln_rate, expl_noise, grad_clip, device, normalizer):

        """
        Implement an ensemble of actors, one for each model and critic ensemble (i.e. 1-to-1 pairing), each Actor is trained with a corresponding model and critic

        Args:
            d_action (int): dimensionality of action
            d_state (int): dimensionality of state
            n_units (int): size or width of hidden layers
            n_layers (int): number of hidden layers (number of non-lineatities). should be >= 2
            ensemble_size (int): number of models in the ensemble
            activation (str): 'linear', 'swish' or 'leaky_relu'
            expl_noise (float): amount of noise to be added to deterministic actions
            device (str): device of the model
        """


        assert n_layers >= 2, "minimum depth of model is 2"

        super().__init__()

        layers = []
        # Initialise list of layers, based on Ensemble layer defined above
        for lyr_idx in range(n_layers + 1):
            if lyr_idx == 0:
                # Note: if using doubleQLearning need to double the ensemble size
                lyr = EnsembleLayer(d_state, n_units, ensemble_size, non_linearity=activation) 
            elif 0 < lyr_idx < n_layers:
                lyr = EnsembleLayer(n_units, n_units, ensemble_size, non_linearity=activation)
            else:  # lyr_idx == n_layers:
                lyr = EnsembleLayer(n_units, d_action, ensemble_size, non_linearity='linear')
            layers.append(lyr)

        self.layers = nn.Sequential(*layers)


        self.to(device)
        self.d_action = d_action
        self.d_state = d_state
        self.n_hidden = n_units
        self.n_layers = n_layers
        self.ensemble_size = ensemble_size
        self.device = device
        self.grad_clip = grad_clip
        self.activation = activation
        self.n_units = n_units
        self.ln_rate = ln_rate

        self.optimiser = opt.Adam(self.parameters(),ln_rate)

    def forward(self, states):

        """
        Predict a set of Q values, one for each ensemble
        Args
            states: [ensemble_size, batch_size, d_state]
        """
        op = self.layers(states)

        return op

    def forward_all(self,states):

        """
        predict action for the same batch of states and actions across all the ensemble.
        takes in raw states and actions and internally normalizes it.
        Args:
            states (torch Tensor[batch size, dim_state])
        Returns:
                action for each ensemble
        """

        states = states.to(self.device)

        # Duplicate each state and action entry for each ensemble
        states = states.unsqueeze(0).repeat(self.ensemble_size, 1, 1)

        actions = self(states)

        return actions


    def random_ensemble(self, states):
        """ Returns a Q for a single model in the ensemble (selected at random), randomised across batches """
        batch_size = states.shape[0]

        # Get Q for all components in the ensemble
        actions = self.forward_all(states)  # shape: (ensemble_size, batch_s, 1)

        i = torch.randint(self.ensemble_size, size=(batch_size,), device=self.device)
        j = torch.arange(batch_size, device=self.device)
        actions = actions[i, j]

        return actions

    def update(self, Q):

        loss = -1 * Q.mean()
        # Optimize the actor
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss
