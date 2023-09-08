import torch as t
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as opt

from torch.distributions import Normal


# Define linear activation
# TODO: should this be a nn Module?
def linear(x):
    return x


# Define swish activation, which may work well for trying Taylor-TD stuff since
# its second derivative is non-zero
@t.jit.script
def swish(x):
    return x * t.sigmoid(x)


# TODO: why are we not using the standard SILU function?
# nn.SiLU()
class Swish(nn.Module):
    def forward(self, input):
        return swish(input)


# Implement ensemble layers to perform a unique forward pass for all models
class EnsembleLayer(nn.Module):
    """An ensemble neural network layer.

    This is like a batched version of a nn.Linear layer, followed by an
    element-wise non-linearity.
    """

    def __init__(self, n_in, n_out, ensemble_size, non_linearity):
        super().__init__()

        weights = t.zeros(ensemble_size, n_in, n_out).float()
        biases = t.zeros(ensemble_size, 1, n_out).float()

        # Initialise weights differently depending on the activation
        # (taken from MAGE)
        #
        # TODO: do we understand why we use the different initialisations
        # depending on the nonlinearity? Short of reading the individual
        # papers, does the pytorch documentation of each of these
        # nonlinearities explain this?
        for weight in weights:
            weight.transpose_(1, 0)

            if non_linearity == 'swish':
                nn.init.xavier_uniform_(weight)
            elif non_linearity == 'leaky_relu':
                nn.init.kaiming_normal_(weight)
            elif non_linearity == 'tanh':
                nn.init.kaiming_normal_(weight)
            elif non_linearity == 'linear':
                nn.init.xavier_normal_(weight)

            weight.transpose_(1, 0)

        # Set model parameters
        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

        # Set the activation functions
        if non_linearity == 'swish':
            self.non_linearity = swish
        elif non_linearity == 'leaky_relu':
            self.non_linearity = F.leaky_relu
        elif non_linearity == 'tanh':
            self.non_linearity = t.tanh
        elif non_linearity == 'linear':
            self.non_linearity = linear

    def forward(self, inp):
        # computation is done using batch matrix multiplication
        # hence forward pass through all models in the ensemble can be done in one call
        return self.non_linearity(
                t.baddbmm(self.biases, inp, self.weights))


class ForwardModel(nn.Module):
    min_log_var = -5
    max_log_var = -1

    def __init__(self, d_action: int, d_state: int, n_units: int,
                 n_layers: int, ensemble_size: int, activation: str,
                 device: str, ln_rate: float, weight_decay: float,
                 grad_clip: float, d_reward, normalizer):
        # TODO: refactor this. Having this many arguments is poor form. Pass an
        # object as a parameter instead.
        # TODO: is predicting the mean and variance of a Gaussian the most
        # effective way to do density estimation? Perhaps parametrising the
        # natural parameters of a Gaussian provides a better conditioned
        # optimisation objective?
        """
        Predicts mean and variance of next state given state and action i.e
        independent gaussians for each dimension of next state, using state and
        action, delta of state is computed.

        The mean of the delta is added to current state to get the mean of next
        state.

        There is a soft threshold on the output variance, forcing it to be in
        the same range as the variance of the training data.

        The thresholds are learnt in the form of bounds on variance and a small
        penalty is used to contract the distance between the lower and upper
        bounds.

        loss components:
            1. minimize negative log-likelihood of data
            2. (small weight) try to contract lower and upper bounds of
               variance

        Args:
            d_action (int): dimensionality of action
            d_state (int): dimensionality of state
            n_units (int): size or width of hidden layers
            n_layers (int): number of hidden layers (number of
                            non-lineatities). should be >= 2
            ensemble_size (int): number of models in the ensemble
            activation (str): 'linear', 'swish' or 'leaky_relu'
            device (str): device of the model
            weight_decay: L2 weight decay on model parameters
            normalizer: class to normalise states, actions and rewards
        """

        assert n_layers >= 2, "minimum depth of model is 2"

        super().__init__()

        layers = []
        # Initialise list of layers, based on Ensemble layer defined above
        for lyr_idx in range(n_layers + 1):
            if lyr_idx == 0:
                lyr = EnsembleLayer(d_action + d_state, n_units, ensemble_size,
                                    non_linearity=activation)
            elif 0 < lyr_idx < n_layers:
                lyr = EnsembleLayer(n_units, n_units, ensemble_size,
                                    non_linearity=activation)
            else:  # output mean and std for both next_state and rwd
                lyr = EnsembleLayer(n_units,  2*d_state + 2*d_reward,
                                    ensemble_size, non_linearity='linear')
            layers.append(lyr)

        self.layers = nn.Sequential(*layers)

        self.normalizer = normalizer

        self.to(device)
        self.d_action = d_action
        self.d_state = d_state
        self.d_rwd = d_reward
        self.n_hidden = n_units
        self.n_layers = n_layers
        self.ensemble_size = ensemble_size
        self.grad_clip = grad_clip
        self.device = device

        # Initialise optimiser
        self.optimiser = opt.Adam(self.parameters(), ln_rate,
                                  weight_decay=weight_decay)

    
    def normalise_model_inputs(self, states, actions):

        states = states.to(self.device)
        actions = actions.to(self.device)

        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
            actions = self.normalizer.normalize_actions(actions)
        return states, actions

    def normalise_rwd(self, rwd: t.Tensor) -> t.Tensor:
        rwd = rwd.to(self.device)

        if self.normalizer is not None:
            rwd = self.normalizer.normalize_rewards(rwd)
        return rwd

    def normalise_model_targets(self, state_deltas):
        state_deltas = state_deltas.to(self.device)

        if self.normalizer is not None:
            state_deltas = self.normalizer.normalize_state_deltas(state_deltas)
        return state_deltas

    def deNormalise_model_outputs(self, delta_mean, rwd_mean, delta_var, rwd_var):

        # denormalize to return in raw state space
        if self.normalizer is not None:
            delta_mean = self.normalizer.denormalize_state_delta_means(delta_mean)
            delta_var = self.normalizer.denormalize_state_delta_vars(delta_var)

            rwd_mean = self.normalizer.denormalize_rewards_mean(rwd_mean)
            rwd_var = self.normalizer.denormalize_rewards_vars(rwd_var)

        return delta_mean, rwd_mean, delta_var, rwd_var

    def forward_pass(self, states, actions):

        inp = torch.cat((states, actions), dim=2)
        op = self.layers(inp)

        # divide prediction into two equal chuncks, one with all predicted mean for each state dim
        # the other with the all predicted std for each state dim 
        overall_mean, log_var = torch.split(op, self.d_state + self.d_rwd, dim=2) 

        # delta: change in state from current to next
        # Extract corresponding mean and var for delta and rwd
        delta_mean = overall_mean[:,:,:-1]
        rwd_mean = overall_mean[:,:,-1:]
        
        delta_logVar = log_var[:,:,:-1]
        rwd_logVar = log_var[:,:,-1:]


        delta_logVar = self.min_log_var + (self.max_log_var - self.min_log_var) * torch.sigmoid(delta_logVar)

        # return mean and var for both change in state (delta) and rwd
        return delta_mean, rwd_mean, delta_logVar.exp(), rwd_logVar.exp()


    # Forward assuming a corresponding state and action for each ensamble
    def forward(self, states, actions):
        """
        Predict next state mean and variance, assuming a entry in state and
        action for each ensemble - i.e., see Args.
        Takes in raw states and actions and internally normalizes it. predict
        rwd mean and variance.

        Args:
            states (torch Tensor[ensemble_size, batch size, dim_state])
            actions (torch Tensor[ensemble_size, batch size, dim_action])
        Returns:
            next state means (torch Tensor[ensemble_size, batch size, dim_state])
            next state variances (torch Tensor[ensemble_size, batch size, dim_state])

        Args:
            states (torch Tensor[ensemble_size, batch size, dim_state])
            actions (torch Tensor[ensemble_size, batch size, dim_action])
        Returns:
            next state means (torch Tensor[ensemble_size, batch size, dim_state])
            next state variances (torch Tensor[ensemble_size, batch size, dim_state])
        """

        # normalise states and actions
        normalized_states, normalized_actions = self.normalise_model_inputs(states, actions)
        # predict change in state (in terms of mean and var in normalised space - since trained with normalised targets, see loss)
        normalized_delta_mean, normalized_rwd_mean, normalized_delta_var, normalized_rwd_var = self.forward_pass(normalized_states, normalized_actions)

        # Denormalise the prediction to obtain raw state space metric
        delta_mean, rwd_mean, delta_var, rwd_var = self.deNormalise_model_outputs(normalized_delta_mean, normalized_rwd_mean, normalized_delta_var, normalized_rwd_var)

        # Predict next state based on current state + predicted_delta in raw state space
        next_state_mean = delta_mean + states

        return next_state_mean, rwd_mean, delta_var, rwd_var

    def forward_all(self, states, actions):
        """
        Predict next state mean and variance of a *batch* of states and actions
        for all models in the ensemble.

    
    def forward_all(self,states,actions):

        Args:
            states (torch Tensor[batch size, dim_state])
            actions (torch Tensor[batch size, dim_action])

        Returns:
            next state means (torch Tensor[batch size, ensemble_size, dim_state])
            rwd means (torch Tensor[batch size, ensemble_size, dim_reward])
            next state variances (torch Tensor[batch size, ensemble_size, dim_state])
            rwd variances (torch Tensor[batch size, ensemble_size, dim_reward])
        """
        states = states.to(self.device)
        actions = actions.to(self.device)

        # Duplicate each state and action entry for each ensemble
        states = states.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        actions = actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1)

        # Predict delta_state in raw state space (see forward())
        next_state_means, rwd_means, next_state_vars, rwd_vars = self(states, actions) # size=(ensemble_s,batch_s,state_d)

        return next_state_means, rwd_means, next_state_vars, rwd_vars

    def sample_SingleEnsemble(self, states, actions):

        """ Returns a distribution for a single model in the ensemble (selected at random) across batch
        Args:
            states: [batch_size, d_state]
            actions: [ batch_size, d_action]
        """

        batch_size = states.shape[0]
        # Get next state distribution for all components in the ensemble
        next_state_means, rwd_means, next_state_vars, rwd_vars = self.forward_all(states, actions)  # shape: [ensemble_size, batch_size, d_state]

        i = t.randint(self.ensemble_size, size=(batch_size,), device=self.device)
        j = t.arange(batch_size, device=self.device)

        # Select a different ensemble prediction for each element in the batch (i.e. j spans over entire batch)
        s_mean = next_state_means[i, j]
        s_var = next_state_vars[i, j]

        r_mean = rwd_means[i, j]
        r_var = rwd_vars[i, j]

        state_pdf = Normal(s_mean, s_var.sqrt())
        rwd_pdf = Normal(r_mean, r_var.sqrt())

        sampled_states = state_pdf.sample() # use rsample() if need re-param trick to pass gradients
        sampled_rwds = rwd_pdf.sample()

        return sampled_states.unsqueeze(0), sampled_rwds.unsqueeze(0)


    def sample_allEnsemble(self, states, actions):
        """ Returns next predicted state given a distribution for each model in the ensemble
        Args:
            states: [ensemble_size, batch_size, d_state]
            actions: [ensemble_size, batch_size, d_action]
        """

        # Get next state distribution for all components in the ensemble
        next_state_means, rwd_means, next_state_vars, rwd_vars = self.forward_all(states, actions)  # shape: (batch_size, ensemble_size, d_state)


        state_pdf = Normal(next_state_means, next_state_vars.sqrt())
        rwd_pdf = Normal(rwd_means, rwd_vars.sqrt())

        sampled_states = state_pdf.sample() # use rsample() if need re-param trick to pass gradients
        sampled_rwds = rwd_pdf.sample()

        return sampled_states, sampled_rwds

    def update(self, states, actions, rwds, state_deltas):

        """
        compute loss given states, actions and state_deltas
        the loss is computed between normalised predicted state delta and actual normalised state delta
        Args:
            states (torch Tensor[ensemble_size, batch size, dim_state])
            actions (torch Tensor[ensemble_size, batch size, dim_action])
            state_deltas (torch Tensor[ensemble_size, batch size, dim_state])
        Returns:
            loss (torch 0-dim Tensor, scalar): `.backward()` can be called on it to compute gradients
        """

        # Normalise transition
        states, actions = self.normalise_model_inputs(states, actions)
        delta_targets = self.normalise_model_targets(state_deltas)
        rwd_targets = self.normalise_rwd(rwds)

        # Compute model prediction
        delta_mu, rwd_mu, delta_var, rwd_var = self.forward_pass(states, actions)      # delta and variance

        # negative log likelihood for both deltas and rewards
        delta_loss = (delta_mu - delta_targets) ** 2 / delta_var + t.log(delta_var)
        rwd_loss = (rwd_mu - rwd_targets) ** 2 / rwd_var + t.log(rwd_var)

        overall_loss = t.mean(delta_loss) + t.mean(rwd_loss)

        self.optimiser.zero_grad()
        overall_loss.backward()
        t.nn.utils.clip_grad_value_(self.parameters(), clip_value=self.grad_clip)
        self.optimiser.step()

        return overall_loss.detach()
