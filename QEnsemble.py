import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

from modelEnsemble import EnsembleLayer


class CriticEnsemble(nn.Module):

   def __init__(self, d_action, d_state, n_units, n_layers, ensemble_size, activation,ln_rate, grad_clip, device, doubleQLearning=True):
        """  
        That is, this implements a 1-to-1 pairing, each Q is trained with a
        corresponding model.

        Args:
            d_action: dimensionality of the actions
            d_state: dimensionality of the state
            # TODO: make list of layer widths
            n_units: width of all hidden layers
            n_layers: number of network layers + activations.
                Should be >= 2
            ensemble_size: number of networks to use in the ensemble
            activation: activation function to use,
                one of 'linear', 'swish', 'leaky_relu'
            ln_rate: learning rate
            grad_clip: gradient normalisation clipping parameter
            device: torch device to use
        """

        assert n_layers >= 2, "minimum depth of model is 2"
        super().__init__()

        if doubleQLearning:
            self.n_Q_nn = 2 # i.e. use two Q networks if using double Q-learning
        else:
            self.n_Q_nn  = 1

        layers = []

        # Initialise list of layers, based on Ensemble layer defined above
        for lyr_idx in range(n_layers + 1):
            if lyr_idx == 0:
                # Note: if using doubleQLearning need to double the ensemble size
                lyr = EnsembleLayer(d_action + d_state, n_units, self.n_Q_nn * ensemble_size, non_linearity=activation) 
            elif 0 < lyr_idx < n_layers:
                lyr = EnsembleLayer(n_units, n_units, self.n_Q_nn * ensemble_size, non_linearity=activation)
            else:  # lyr_idx == n_layers:
                lyr = EnsembleLayer(n_units, 1, self.n_Q_nn * ensemble_size, non_linearity='linear')
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

    def forward(self, states: t.Tensor, actions: t.Tensor) -> t.Tensor:
        """
        Predict a set of Q values, one for each ensemble
        Args
            states: [ensemble_size, batch_size, d_state]
            actions: [ensemble_size, batch_size, d_state]
        """

        inp = t.cat((states, actions), dim=2)
        op = self.layers(inp)

        return op

    def forward_all(self, states: t.Tensor, actions: t.Tensor) -> t.Tensor:
        """
        predict Qs for the same batch of states and actions across all the
        ensemble.

        takes in raw states and actions and internally normalizes it.
        Args:
            states (torch Tensor[batch size, dim_state])
            actions (torch Tensor[batch size, dim_action])
        Returns:
                Q prediction for each ensemble
        """

        states = states.to(self.device)
        actions = actions.to(self.device)

        # Duplicate each state and action entry for each ensemble
        states = states.unsqueeze(0).repeat(self.n_Q_nn * self.ensemble_size, 1, 1)
        actions = actions.unsqueeze(0).repeat(self.n_Q_nn * self.ensemble_size, 1, 1)

        Qs = self(states,actions)

        return Qs

    def random_ensemble(self, states: t.Tensor, actions: t.Tensor) -> t.Tensor:
        """Returns a Q for a single model in the ensemble (selected at random).

        Randomises across batches

        Args:
            states: a batch of states [batch size, state_dim]
            actions: a batch of actions [batch, action_dim]

        Returns:
            t.Tensor: the random value function's output
        """

        batch_size = states.shape[0]

        # Get Q for all components in the ensemble
        # shape: (ensemble_size, batch_s, 1)
        Qs = self.forward_all(states, actions)

        i = t.randint(self.ensemble_size, size=(batch_size,),
                      device=self.device)
        j = t.arange(batch_size, device=self.device)
        Qs = Qs[i, j]

        return Qs

    def min_ensemble(self, states: t.Tensor, actions: t.Tensor) -> t.Tensor:
        """Returns the lowest Q across the ensemble, for each element in the
        batch

        Args:
            states: batch of states
            actions: batch of actions
        """

        # Get Q for all components in the ensemble
        # shape: (ensemble_size, batch_s, 1)
        Qs = self.forward_all(states, actions)
        min_Qs, _ = Qs.min(dim=0)

        return min_Qs

    def update_Qs(self, td_errors: t.Tensor) -> t.Tensor:
        """
        Update Q functions
        Args:
            td_errors: used to optimise critic [ensemble_size, batch_s, 1]
        """

        # loss = torch.mean((targets - predictions)**2)
        zero_targets = t.zeros_like(td_errors, device=td_errors.device)
        # use huber loss, worked well in MAGE
        loss = 0.5 * F.smooth_l1_loss(td_errors, zero_targets)

        self.optimiser.zero_grad()
        loss.backward()
        t.nn.utils.clip_grad_value_(self.parameters(), self.grad_clip)
        self.optimiser.step()

        return loss
