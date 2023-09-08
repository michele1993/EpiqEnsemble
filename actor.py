import torch as t
import torch.nn as nn
import torch.optim as opt
from modelEnsemble import Swish


class Actor(nn.Module):

    def __init__(self, d_action: int, d_state: int, n_units: int,
                 n_layers: int, activation: str, ln_rate: float,
                 expl_noise: float, grad_clip: float, device: str,
                 normalizer):
        """Generic policy netwok

        Args:
            d_action: dimensionality of the actions
            d_state: dimensionality of the states
            n_units: width of MLP hidden layers
            n_layers: number of layers + activations
            activation: activation function to use
            ln_rate: policy network learning rate
            expl_noise: amount of dithering for exploration
            grad_clip: gradient norm clipping
            device: torch device to use
            normalizer: ??
        """
        super().__init__()

        self.expl_noise = expl_noise
        self.normalizer = normalizer
        self.grad_clip = grad_clip
        self.device = device

        assert n_layers >= 1

        layers = [nn.Linear(d_state, n_units), self.get_activation(activation)]
        for _ in range(1, n_layers):
            layers += [nn.Linear(n_units, n_units),
                       self.get_activation(activation)]
        layers += [nn.Linear(n_units, d_action)]

        [self.init_weights(layer) for layer in layers
         if isinstance(layer, nn.Linear)]

        self.layers = nn.Sequential(*layers)

        self.to(device)

        self.optimizer = opt.Adam(self.parameters(), ln_rate)

    def init_weights(self, layer: nn.Module) -> None:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0)

    def get_activation(self, activation: str) -> nn.Module:
        """Given the string-valued activation function, return the PyTorch
        module impementing it.

        TODO: make this Actor class accept an un-initialised pytorch activation
        module.
        """
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
        return t.tanh(self.layers(state))  # Bound to -1,1

    # Return a exploratory action without gradient
    def expl_action(self, state: t.Tensor) -> t.Tensor:

        if self.normalizer is not None:
            state = self.normalizer.normalize_states(state)

        det_a = self(state)
        noise = t.randn_like(det_a, device=self.device) * self.expl_noise
        return  (det_a.detach() + noise).clamp(-1,1)  # actions bounded [-1,1]

        return  (det_a + torch.randn_like(det_a, device=self.device) * self.expl_noise).clamp(-1,1) # actions bounded [-1,1]

        Args:
            Q: the action-value functions

        Returns:
            t.Tensor: the policy loss
        """
        loss = -1 * Q.mean()
        # Optimize the actor
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss
