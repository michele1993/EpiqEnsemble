import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from actor import Actor 
from QEnsemble import CriticEnsemble 

class TD3():
     def __init__(
            self,
            d_action,
            d_state,
            actor,
            critic,
            fm_ensemble_s,
            q_ensemble_s,
            batch_s,
            device,
            gamma,
            normalizer,
            tau=0.005,
            policy_delay=2,
            policy_smt_noise=0.2,
            noise_clip=0.5
    ):
        super().__init__()

        self.d_action = d_action
        self.d_state = d_state
        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor)
        self.freeze_params(self.actor_target) # ensure target net does not require grad
        
        self.fm_ensemble_s = fm_ensemble_s
        self.q_ensemble_s = q_ensemble_s
        self.batch_s = batch_s

        self.discount = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.policy_smt_noise = policy_smt_noise
        self.noise_clip = noise_clip
        self.device = device

        self.step_counter = 0

        self.normalizer = normalizer

        # Initialise two critics with two targets
        self.critic = critic
        self.critic_target = CriticEnsemble(d_action, d_state, critic.n_units, critic.n_layers, 2*q_ensemble_s, critic.activation, critic.ln_rate, critic.grad_clip, self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.freeze_params(self.critic_target) # ensure target net does not require grad


     def freeze_params(self, network): # detach gradient of target networks

        for params in network.parameters():

             params.requires_grad = False


     def update(self, states, actions, rewards, next_states, masks):
        """
       Update the policy and the critic through the TD3 approach, with a slight variation, a single criti target net is used without the min() operation for simplicity

       Args:
           states: current states [batch_s, d_state]
           actions: current actions, [batch_s, d_action]
           rewards: predicted rewards [ensemble_s, batch_s, 1]
           next_states: predicted next_states [ensemble_s, batch_s, d_state]
           masks: determine the done condition [ensemble_s, batch_s, 1]
        """

        # if normalizer is set-up, normalise states
        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
            next_states = self.normalizer.normalize_states(next_states)
        self.step_counter += 1

        # Reshape next_states to pass to policy network, you need a different target action for each next_state predicted by each model in the ensemble
        next_states_pol = next_states.view(self.q_ensemble_s * self.batch_s, self.d_state)

        # Select target action according to policy and add clipped noise
        det_next_actions = self.actor_target(next_states_pol)
        noise = (
                torch.randn_like(det_next_actions) * self.policy_smt_noise
        ).clamp(-self.noise_clip, self.noise_clip)

        
        # Reshape actions in format to be passed to Q ensemble
        next_actions = (det_next_actions + noise).clamp(-1, 1).view(self.q_ensemble_s,self.batch_s,self.d_action)


        # Duplicate states and actions to pass to two target ensembles
        next_states = next_states.repeat(2, 1, 1)
        next_actions = next_actions.repeat(2, 1, 1)

        # Compute the min target Q value across two target ensembles
        next_Qs = self.critic_target(next_states, next_actions) # Q has shape [2*ensable_s, batch_s, 1]
        next_Qs1, next_Qs2 = torch.split(next_Qs, self.q_ensemble_s, dim=0) # divide predictions from two targets Q ensembles
        q_targets = torch.min(next_Qs1, next_Qs2)

        # Add the mask(~done) and reward
        targets = rewards + self.discount * (~masks).float() * q_targets # Note: invert the mask! the rwd is already normalised since predicted by the model

        # Get current Q estimates for two ensebles
        current_Qs = self.critic.forward_all(states, actions.detach()) # predict Q for same current (s,a) across ensemble
        current_Qs1, current_Qs2 = torch.split(current_Qs, self.q_ensemble_s, dim=0) # divide predictions from two Q ensembles
        td_errors_1, td_errors_2 = targets - current_Qs1, targets - current_Qs2 # compute td error for each ensemble

        td_errors = torch.cat([td_errors_1, td_errors_2], dim=0) # put td erros together to train both Q ensembles

        # Optimize the critic:
        critic_loss = self.critic.update_Qs(td_errors)


        # Optimize the policy:
        if self.step_counter % self.policy_delay == 0:

            det_a = self.actor(states)
            # Compute actor loss
            q = self.critic.forward_all(states, det_a)  
            q1, q2 = torch.split(q, self.q_ensemble_s, dim=0) # divide predictions from two targets Q ensembles
            targ_q = torch.min(q1,q2)

            actor_loss = self.actor.update(targ_q).detach()
            self.last_actor_loss = actor_loss.item()

            # Update the frozen target policy
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update the frozen target value function
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
