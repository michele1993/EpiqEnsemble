import copy
import torch
from modelEnsemble import ForwardModel
from QEnsemble import CriticEnsemble
from QHeadEnsmble import CriticHeadEnsemble
from memoryBuffer import Buffer
from td3 import TD3
from unc_td3 import unc_TD3
from actor import Actor
import numpy as np
from utils import to_torch, to_np
from normalizer import TransitionNormalizer
import logging
import gym
import envs.gymmb  # Register custom gym envs
from wrappers import BoundedActionsEnv, IsDoneEnv, MuJoCoCloseFixWrapper

class TrainingLoop():

    def __init__(self,
              n_total_steps,
              n_warmup_steps,
              ensemble_size,   
              exploratory_steps,
              n_dyna_transitions,  
              env_name,
              device,
              buffer_s,
              batch_s, 
              fm_model_batch_s,
              fm_n_model_training_iter,
              fm_model_training_freq,
              fm_n_units,
              fm_n_layers,
              fm_activation,
              fm_lr,
              q_n_units,
              q_n_layers,
              q_activation,
              q_lr,
              a_n_units,
              a_n_layers,
              a_activation,
              a_lr,
              expl_noise,
              alg_name='unc_td3',
              doubleQLearning=True,
              gamma=0.99, 
              grad_clip=5,
              normalize_transitions=False,
              d_reward=1,
              full_ensemble=True
              ):

        if alg_name =='unc_td3' or alg_name =='td3':
            assert doubleQLearning , 'must use double Q-learning for the current algorithm'
              
        self.device = device
        self.n_total_steps = n_total_steps
        self.n_warmup_steps = n_warmup_steps # n steps before starting to update agent
        self.exploratory_steps = exploratory_steps # n steps using uniform random actions

        self.env_name = env_name
        self.trn_env = self._setup_env(env_name) # initialise the training env
        self.c_env = self.trn_env # set the training env as current env, this is done as we also use evaluation env

        self.d_action = self.trn_env.action_space.shape[0]
        self.d_state = self.trn_env.observation_space.shape[0]
        self.d_reward = d_reward
        self.gamma = gamma
        ## Key: set all ensembles to the same value for 1-to-1 pairing 
        self.fm_ensemble_s = ensemble_size
        self.q_ensemble_s = ensemble_size
        self.act_ensemble_s = ensemble_size
        self.batch_s = batch_s
        self.model_batch_s = fm_model_batch_s
        self.n_dyna_transitions = n_dyna_transitions
        self.model_training_freq = fm_model_training_freq        
        self.n_model_training_iter = fm_n_model_training_iter
        self.actor = None


        logging.info(f'Env: {env_name}, Alg: {alg_name}, N steps: {n_total_steps}, Q_lr: {q_lr}, Actor_lr: {a_lr}, Explor Noise: {expl_noise}, Ensemble size: {ensemble_size}, Full ensemble: {full_ensemble}')

        if normalize_transitions:
            self.normalizer = TransitionNormalizer(self.d_state,self.d_action,self.device)
        else:
            self.normalizer= None

        self.m_buffer = self._setup_Buffer(size=buffer_s)

        self.actor = self._setup_Actor(n_units=a_n_units, n_layers=a_n_layers, ensemble_size=act_ensemble_s, activation=a_activation, ln_rate=a_lr, exploration_noise=expl_noise, grad_clip=grad_clip)

        self.critic = self._setup_Critic(n_units=q_n_units, n_layers=q_n_layers, activation=q_activation, ln_rate=q_lr, grad_clip=grad_clip, doubleQLearning=doubleQLearning, full_ensemble=full_ensemble)

        self.f_model = self._setup_FModel(n_units=fm_n_units, n_layers=fm_n_layers, ensemble_size=fm_ensemble_s, activation=fm_activation, ln_rate=fm_lr,grad_clip=grad_clip)
              
        self.alg = self._setup_alg(alg_name, full_ensemble)
                   
    def _setup_env(self, env_name):
        # ------ Setup the environment with wrappers----
        env = gym.make(env_name)
        env = BoundedActionsEnv(env) # bound all actions between [-1,1]
        env = MuJoCoCloseFixWrapper(env) # easier to close mujoco

        return env

    def _setup_alg(self, alg_name, full_ensemble):

        assert self.actor is not None and self.critic is not None, 'Need to setup Actor and Critic first'

        if alg_name =='td3':
            self.sample_single_ensemble = True 
            self.q_actor = self.actor # in td3 the behavioural policy = target policy (in terms of parameters)
            return TD3(d_action=self.d_action, d_state=self.d_state, actor=self.actor, critic=self.critic, fm_ensemble_s=self.fm_ensemble_s, q_ensemble_s=self.q_ensemble_s, batch_s=self.batch_s, device=self.device, gamma=self.gamma, normalizer=self.normalizer)

        elif alg_name =='unc_td3':
            self.sample_single_ensemble = False # want a Q for each ensemble
            self.q_actor = copy.deepcopy(self.actor) # initialise a separate actor to be trained differently than target actor
            return unc_TD3(d_action=self.d_action, d_state=self.d_state, actor=self.actor, q_actor=self.q_actor, critic=self.critic, fm_ensemble_s=self.fm_ensemble_s, q_ensemble_s=self.q_ensemble_s, batch_s=self.batch_s, device=self.device, gamma=self.gamma, normalizer=self.normalizer, full_ensemble=full_ensemble)
            
        else:
            raise NotImplementedError('Algorithm not implemented')

    def _setup_FModel(self, n_units, n_layers, ensemble_size, activation, ln_rate, grad_clip, weight_decay=1e-4):

        return ForwardModel(self.d_action, self.d_state,n_units, n_layers, ensemble_size, activation, self.device, ln_rate, weight_decay, grad_clip,  self.d_reward,self.normalizer)

    def _setup_Actor(self, n_units, n_layers, ensemble_size, activation, ln_rate, exploration_noise, grad_clip):

        return ActorEnsemble(self.d_action, self.d_state, n_units, n_layers, ensemble_size, activation, ln_rate, exploration_noise, grad_clip, self.device, self.normalizer)

    def _setup_Critic(self, n_units, n_layers, activation, ln_rate, grad_clip, doubleQLearning, full_ensemble):

        if full_ensemble:
            return CriticEnsemble(self.d_action, self.d_state, n_units, n_layers, self.q_ensemble_s, activation, ln_rate, grad_clip, self.device, doubleQLearning)
        else:
            return CriticHeadEnsemble(self.d_action, self.d_state, n_units, n_layers, self.q_ensemble_s, activation, ln_rate, grad_clip, self.device,doubleQLearning)

    def _setup_Buffer(self, size):
        return Buffer(self.d_action, self.d_state, size, self.normalizer, self.device, self.d_reward) 

    # take a step in the environment and return torch.tensor instead of np.array
    def _env_step(self, action):
            n_state, rwd, done, _ = self.c_env.step(to_np(action))
            return to_torch(n_state).to(self.device), to_torch(np.array([rwd], dtype=np.float)), done

    def _env_reset(self):    
        return to_torch(self.c_env.reset()).to(self.device)

    def log_stats(self,s, rwd, done, n_eps=5, eval_freq=1000): # Note: for env with more that 1000 steps print same accuracy more than once
        """
        store and log some useful training statistics
        """
        
        self.ep_acc.append(rwd)
        self.t+=1

        if done:
            mean_acc = (sum(self.ep_acc)).item()
            mean_step = self.t
            #print('Step: ',s)
            #print(f'Last {n_eps} episodes acc ', mean_acc,'\n')
            logging.info(f'| Step: {s} | Training mean accuracy: {mean_acc} | Training mean n. steps: {mean_step}')
            self.eps_acc.append(sum(self.ep_acc))
            self.ep_acc = []
            self.t = 0

        # Evaluate agent based on n full episodes with target policy
        if s % eval_freq == 0:
            eval_mean, _, eval_steps = self.evaluate_pol()
            logging.info(f'| Step: {s} | Evaluation mean accuracy: {eval_mean} | Evaluation mean n. steps: {eval_steps} \n')
            self.eval_acc.append(eval_mean)

       
    def imagination(self, states):
        """
        Generate 1-step imaginary transitions given  starting (real) states
        """

        actions = self.q_actor.expl_action(states) # use q_actor to learn a broad Q

        if self.sample_single_ensemble: # sample a prediction from a single ensemble, i.e. return 1 predicted next state
            p_states, p_rwds =  self.f_model.sample_SingleEnsemble(states,actions) 

        else: # sample a prediction from each model in the ensemble, i.e. return n. predicted next states
            p_states, p_rwds = self.f_model.sample_allEnsemble(states,actions) 


        # -------- Uncomment to use true rwd function --------
        #try:
        #    p_rwds = self.trn_env.reward(states, actions,p_states).unsqueeze(0).unsqueeze(-1) 
        #except AttributeError:
        #    print('reward() method not implemented for the current env')
        # --------------------------

        dones = self.c_env.is_done(p_states)

        return actions, p_rwds, p_states, dones

    def train_model(self):
        losses = []
        # divide all elements in the buffer in random batches and iterate through them to update model
        for states, actions, rwds, state_deltas in self.m_buffer.sample_ensemble_batches(self.fm_ensemble_s, self.model_batch_s):
            fm_loss = self.f_model.update(states, actions, rwds, state_deltas) # Normalisation occurs inside f_model
            losses.append(fm_loss.item())
        
        return losses

    def train(self):
        """
        Main training loop
        """


        self.eps_acc = [] # return for each training episodes
        self.eval_acc = [] # evaluation accuracy
        self.ep_acc = [] # reward within an episode
        self.t = 0 # t-steps within each episode

        c_state = self._env_reset()
        #c_state = self.trn_env.reset()

        for s in range(1,self.n_total_steps):

            # --------- interact with the environment and add transitions to the buffer -------
            if s < self.exploratory_steps: # take random exploratory actions for first few steps

                action = torch.rand(size=(1,self.d_action)) * 2 - 1 # set range action range to [-1,1]
            else:
                action = self.actor.expl_action(c_state.to(self.device)).detach() # use target actor plus noise for exploration

            n_state, rwd, done = self._env_step(action)
            #n_state, rwd, done = self.trn_env.step(action)
            self.m_buffer.add(c_state, action, rwd, n_state) # store everything as a tensor and un-normalised

            if not done:
                c_state = n_state
            else:
                c_state = self._env_reset()

            #c_state = self.termination_check(n_state, done)
            self.log_stats(s, rwd, done)

            # ------- Train the transition ensemble ----------
            if s % self.model_training_freq ==0:
                n_i = 0
                # Update model n times independetly of the n of elements in the buffer (i.e. loop adjusted for n iter in inner loop in train_model())
                while n_i < self.n_model_training_iter: 
                    losses = self.train_model()
                    n_i += len(losses)

            # ---------- Train the agent using dyna approach -----------
            if s > self.n_warmup_steps:
                for i in range(self.n_dyna_transitions):
                    states, _, = self.m_buffer.sample_batches(self.batch_s) # Sample a new random batch of states 
                    actions, p_rwds, p_states, p_dones = self.imagination(states) # generate an 'imagined' transition 'on-policy' based on true states
                    self.alg.update(states, actions, p_rwds, p_states, p_dones) #NOTE: it is key that p_rwds, dones are in the same shape as the predicted Q

        return self.eval_acc    

    def evaluate_pol(self,evaluate_eps=5):
        """
        Evaluate agent target policy based on n. full episodes        
        """

        eval_rwd = []
        eval_steps = []

        for ep in range(1,evaluate_eps+1):

            ep_rwd = []
            t=0

            # Create a new env for each evaluation (so that can also set different seeds)
            # and set it as the current environment
            self.c_env = self._setup_env(self.env_name)
            c_state = self._env_reset()
            done = False

            while not done:

                if self.normalizer is not None:
                    c_state = self.normalizer.normalize_states(c_state)

                trgt_action = self.actor(c_state).detach() # take action from target policy
                n_state, rwd, done = self._env_step(trgt_action)
                c_state = n_state

                #c_state = self.termination_check(n_state, done)
                ep_rwd.append(rwd.item())
                t+=1

            eval_rwd.append(sum(ep_rwd))
            eval_steps.append(t)
            self.c_env.close()
        
        # reset training env as the current env
        self.c_env = self.trn_env

        return sum(eval_rwd)/evaluate_eps, None, np.mean(eval_steps)


