import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from modelEnsemble import EnsembleLayer 
import torch.optim as opt



class CriticHeadEnsemble(nn.Module):

    def __init__(self, d_action, d_state, n_units, n_standard_layers, ensemble_size, activation,ln_rate, grad_clip, device, doubleQLearning=True, n_ensb_layers=1):

        """
        Implement an ensemble of critic functions, where you can flexibly specify how many ensemble layers you want (e.g. only the head)
        while the rest being a shared body.
        Note: With the current implementation the input layer cannot be a full ensemble layer, but it is always shared

        Args:
            d_action (int): dimensionality of action
            d_state (int): dimensionality of state
            n_units (int): size or width of hidden layers
            n_standard_layers (int): number of hidden shared layers (number of non-lineatities) for the shared body.
            ensemble_size (int): number of models in the ensemble
            activation (str): 'linear', 'swish' or 'leaky_relu'
            device (str): device of the model
            n_ensb_layers (int): number of ensemble layers added on top of shared body 
        """


        super().__init__()

        if doubleQLearning:
            self.n_Q_nn = 2 # i.e. use two Q networks if using double Q-learning
        else:
            self.n_Q_nn  = 1
        

        assert n_ensb_layers >= 1, "minimum depth of ensemble layers is 1"
        assert n_standard_layers >= 1, "minimum depth of standard layers is 1"

        standard_layers = []

        # Add input layer, Note: with the current implementation the input layer cannot be a full ensemble
        standard_layers.append(EnsembleLayer(d_action + d_state, n_units, self.n_Q_nn, non_linearity=activation))
        
        # Initialise list of non-ensemble hidden layers, we still use EnsembleLayer to implement doubleQ net
        for lyr_idx in range(n_standard_layers-1):
            lyr = EnsembleLayer(n_units, n_units, self.n_Q_nn, non_linearity=activation)
            standard_layers.append(lyr)
        self.standard_layers = nn.Sequential(*standard_layers)

        ensb_layers = []

        # Initialise list of ensemble hidden layers
        for lyr_indx in range(n_ensb_layers-1): # -1 because you're adding ouput layer after (so if layers=1, then only have output layer)
            lyr = EnsembleLayer(n_units, n_units, self.n_Q_nn * ensemble_size, non_linearity=activation)
            ensb_layers.append(lyr)
        
        # Add output layer
        ensb_layers.append(EnsembleLayer(n_units, 1, self.n_Q_nn * ensemble_size, non_linearity='linear'))
        self.ensb_layers = nn.Sequential(*ensb_layers)

        self.to(device)
        self.d_action = d_action
        self.d_state = d_state
        self.n_hidden = n_units
        self.n_standard_layers = n_standard_layers
        self.ensemble_size = ensemble_size
        self.device = device
        self.grad_clip = grad_clip
        self.activation = activation
        self.n_units = n_units
        self.ln_rate = ln_rate
        self.doubleQLearning = doubleQLearning
        self.n_ensb_layers = n_ensb_layers
        self.optimiser = opt.Adam(self.parameters(),ln_rate)

    def forward(self, states, actions):

        """
        Predict a set of Q values, one for each ensemble across the two Q net (if doubleQ is True)
        Args
            states: [n_Q_nn * ensemble_size, batch_size, d_state]
            actions: [n_Q_nn * ensemble_size, batch_size, d_state]
        Returns:
            Q prediction: [n_Q_nn * ensemble_size, batch_size, 1]
        """

        batch_s = states.size()[1]

        inp = torch.cat((states, actions), dim=2).reshape(self.n_Q_nn, batch_s * self.ensemble_size, self.d_action + self.d_state)

        op = self.standard_layers(inp)
        
        op = self.ensb_layers(op.reshape(self.n_Q_nn * self.ensemble_size, batch_s, self.n_units))

        return op

    def forward_all(self,states,actions):

        """
        predict Qs for the same batch of states and actions across all the ensemble.
        takes in raw states and actions and internally normalizes it.
        Args:
            states (torch Tensor[batch size, dim_state])
            actions (torch Tensor[batch size, dim_action])
        Returns:
                Q prediction for each model in the ensemble
        """


        states = states.to(self.device)
        actions = actions.to(self.device)

        # Duplicate each state and action entry for each ensemble
        states = states.unsqueeze(0).repeat(self.ensemble_size * self.n_Q_nn, 1, 1)
        actions = actions.unsqueeze(0).repeat(self.ensemble_size * self.n_Q_nn, 1, 1)


        Qs = self(states,actions)

        return Qs


    def random_ensemble(self, states, actions):
        """ Returns a Q for a single model in the ensemble (selected at random), randomised across batches """
        batch_size = states.shape[0]

        # Get Q for all components in the ensemble
        Qs = self.forward_all(states, actions)  # shape: (ensemble_size, batch_s, 1)

        i = torch.randint(self.ensemble_size, size=(batch_size,), device=self.device)
        j = torch.arange(batch_size, device=self.device)
        Qs = Qs[i, j]

        return Qs


    def min_ensemble(self, states, actions):
        """ Returns the lowest Q across the ensemble, for each element in the batch """
        batch_size = states.shape[0]

        # Get Q for all components in the ensemble
        Qs = self.forward_all(states, actions)  # shape: (ensemble_size, batch_s, 1)
        min_Qs, _ = Qs.min(dim=0)

        return min_Qs


    def update_Qs(self, td_errors): 

        """
        Update Q functions
        Args:
            td_errors: used to optimise critic [ensemble_size, batch_s, 1]
        """

        #loss = torch.mean((targets - predictions)**2)
        zero_targets = torch.zeros_like(td_errors, device=td_errors.device)
        loss = 0.5 * F.smooth_l1_loss(td_errors, zero_targets) # use huber loss, worked well in MAGE

        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), self.grad_clip) 
        self.optimiser.step()

        return loss
