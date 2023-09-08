"""
TODO: consolidate this into a single module (use the other, tested buffer as a
reference)

Design a consistent API that satisfies all the use-cases.
"""

import warnings
import torch as t
import numpy as np

from epiq.utils import to_torch


# NOTE: initally I wanted to store normalised states, performing the
# normalisation within the buffer, then I realised this is not a great idea
# since the normalisation values will change over training and don't want old
# states to have the "wrong" normalisation

class Buffer:
    def __init__(self, d_action: int, d_state: int, size: int,
                 normalizer, device, d_reward):
        """
        data buffer that holds transitions
        Args:
            d_state: dimensionality of state
            d_action: dimensionality of action
            size: maximum number of transitions to be stored (memory allocated
                at init)
        """

        # Dimensions
        self.size = size
        self.d_state = d_state
        self.d_action = d_action
        self.d_reward = d_reward
        self.device = device

        # Main Attributes: define a buffer for each transition entry
        self.states = t.zeros(size, d_state).float()
        self.actions = t.zeros(size, d_action).float()
        self.rewards = t.zeros(size, d_reward).float()
        self.state_deltas = t.zeros(size, d_state).float()

        # Other attributes
        self.normalizer = normalizer
        self.ptr = 0
        self.is_full = False

    # Helper method for actual add() method, ensure buffer re-written when full
    def _add(self, buffer, arr):
        """
        Args:
            buffer: the buffer entry we want to add elements to (e.g.
                state_buffer, action_buffer, reward_buffer etc)
            arr: the elements we want to add to that buffer entry (e.g. states,
                actions, etc.)

        By passing buffer as an entry we can re-use this method across
        state_buffer, action_buffer etc.
        """

        n = arr.size(0)
        excess = self.ptr + n - self.size  # by how many elements we exceed the size
        if excess <= 0:  # all elements fit
            a, b = n, 0
        else:
            a, b = n - excess, excess  # we need to split into a + b = n; a at the end and the rest in the beginning
        buffer[self.ptr:self.ptr + a] = arr[:a]
        buffer[:b] = arr[a:] # Note: if b=0 and a=n no entries will be added

    def add(self, states, actions, rewards, next_states):
        """
        Add a transition to the replay buffer.

        Args:
            states: pytorch Tensors of (n_transitions, d_state) shape
            actions: pytorch Tensors of (n_transitions, d_action) shape
            rewards: pytorch Tensor of (n_transitions, d_rwd) shape
            next_states: pytorch Tensors of (n_transitions, d_state) shape
        """
        # NOTE: The way in which this buffer is implemented constrains the type
        # of [Gym] environments that we can support. Namely, we support only
        # the subset whose state space can be serialised into a 1-dimensional
        # vector.

        # TODO: use a torch environment wrapper which will convert all the
        # numpy state / action arrays into tensors?
        states, actions, rewards, next_states = \
                [x.clone().cpu() for x in [states, actions, rewards, next_states]]
        #states, actions, rewards, next_states = [to_torch(x) for x in [states, actions, rewards, next_states]]

        state_deltas = next_states - states
        n_transitions = states.size(0)

        # Add transitions to normaliser to compute normalisation factors
        if self.normalizer is not None:
            for s, a, ns, r in zip(states, actions, state_deltas, rewards):
                self.normalizer.add(s, a, ns, r)

        assert n_transitions <= self.size # ensure transition size is not bigger than overall buffer size

        # User helper method defined above to add each transition entry to the corresponding buffer
        self._add(self.states, states)
        self._add(self.actions, actions)
        self._add(self.rewards, rewards)
        self._add(self.state_deltas, state_deltas)

        if self.ptr + n_transitions > self.size or self.is_full:
            warnings.warn("Buffer overflow. Rewriting old samples")

        # check if buffer is full
        if self.ptr + n_transitions >= self.size:
            self.is_full = True # leave as True to ensure all elements of buffer considered in len(), since self.ptr get re-set to zero, when buffer full

        self.ptr = (self.ptr + n_transitions) % self.size # count elements in buffer, and re-set when buffer full (i.e. % self.size)


    def sample_batches(self, batch_size):

        """
        return a batch of state and action transitions
        this method is used to get a different Q function and next state for the same state action pairs (across the ensemble)
        """

        batch_idx = t.randint(len(self), size=[batch_size]) # use overridden len() method to get n of elements

        s = self.states[batch_idx].to(self.device)
        a = self.actions[batch_idx].to(self.device)
        #s_delta = self.state_deltas[batch_idx].to(self.device)
        #ns = s + s_delta

        return s, a #, ns, s_delta

    def sample_ensemble_batches(self, ensemble_size, batch_size):
        """
        return an iterator of batches for the ensemble, where a different transition is sampled for ensemble_size x batch_size
        this method can be used to train the model (and Q ensemble ?), enabling the ensable to be trained with different transitions
        Args:
            batch_size: number of samples to be returned
            ensemble_size: size of the ensemble
        Returns:
            state of size (ensemble_size, n_samples, d_state)
            action of size (ensemble_size, n_samples, d_action)
            next state of size (ensemble_size, n_samples, d_state)
        """

        num = len(self)
        indices = [np.random.permutation(range(num)) for _ in range(ensemble_size)] # generate a different indx permuation for each ensemble
        indices = np.stack(indices)

        for i in range(0, num, batch_size): # iterate across indx permutation, by taking a batch (of indxes) at the time
            j = min(num, i + batch_size)

            if (j - i) < batch_size and i != 0:
                # drop the last incomplete batch
                return

            batch_size = j - i

            batch_indices = indices[:, i:j] # select indexes in range across all esemble indixes
            batch_indices = batch_indices.flatten()

            states = self.states[batch_indices]
            actions = self.actions[batch_indices]
            rewards = self.rewards[batch_indices]
            state_deltas = self.state_deltas[batch_indices]

            states = states.reshape(ensemble_size, batch_size, self.d_state)
            actions = actions.reshape(ensemble_size, batch_size, self.d_action)
            rewards = rewards.reshape(ensemble_size, batch_size, self.d_reward)
            state_deltas = state_deltas.reshape(ensemble_size, batch_size, self.d_state)

            yield states.to(self.device), actions.to(self.device), rewards.to(self.device), state_deltas.to(self.device)

    def view(self):

        n = len(self)

        s = self.states[:n]
        a = self.actions[:n]
        s_delta = self.state_deltas[:n]
        r = self.rewards[:n]
        ns = s + s_delta

        return s.to(self.device),a.to(self.device),r.to(self.device),ns.to(self.device)

    def __len__(self):
        return self.size if self.is_full else self.ptr # if full take all element, since self.ptr get re-set to 0
