import os
import logging
import torch
import numpy as np
from modelEnsemble import ForwardModel
from QEnsemble import CriticEnsemble
from memoryBuffer import Buffer
from training_loop import TrainingLoop
from td3 import TD3
import argparse
import gym
import envs.gymmb  # Register custom gym envs
from utils import setup_logger, boolean_string



# Extract input variables:
parser = argparse.ArgumentParser()
parser.add_argument('--env-name','-e',type=str,nargs='?')
parser.add_argument('--q-lr','-qlr',type=float,nargs='?', default=1e-4)
parser.add_argument('--a-lr','-alr',type=float,nargs='?', default=1e-4)
parser.add_argument('--n-steps','-ns',type=int,nargs='?', default=20000)
parser.add_argument('--n-warmups','-nw',type=int,nargs='?', default=1000)
parser.add_argument('--expl-noise','-en',type=float,nargs='?', default=0.1)
parser.add_argument('--full-ensemb','-fe', type=boolean_string, nargs='?', default=True)
parser.add_argument('--seed', '-s', type=int, nargs='?', default=None)
parser.add_argument('--save', '-sv', type=boolean_string, nargs='?', default=False)
parser.add_argument('--run-Type', '-rt', type=str, nargs='?', default='Trial')
parser.add_argument('--run-Number', '-rn', type=int, nargs='?', default=00)

# ---------- Extract input arguments ---------
args = parser.parse_args()

env_name = args.env_name
q_lr = args.q_lr
a_lr = args.a_lr
n_total_steps = args.n_steps
n_warmup_steps = args.n_warmups # n of steps before updating agent
expl_noise = args.expl_noise
seed = args.seed
run_type = args.run_Type
run_number = args.run_Number
save_results = args.save
full_q_ensemble = args.full_ensemb

# -------- Variables ------------------
exploratory_steps = n_warmup_steps # n of steps using a random uniform policy
buffer_s = n_total_steps
batch_s = 1024
alg_name = 'unc_td3'
n_dyna_transitions = 10 # n of 1-step (model) generated transitions
normalize_transitions = True

# Forward model:
fm_n_units = 512
fm_n_layers = 4
fm_ensemble_s = 8
fm_activation = 'swish'
fm_lr = 1e-4
fm_model_training_freq = 25 # how often model updated (in terms of steps in real env)
fm_model_batch_s = 256 # batch size for each model update
fm_n_model_training_iter = 120 # n of model update every time the model is updated

# Q model:
q_n_units = 400
q_n_layers = 3 #2
q_ensemble_s = fm_ensemble_s
q_activation = 'swish'

# Actor:
a_n_units = 400
a_n_layers = 2
a_activation = 'swish'

# ----- Setup directory for saving results --------
if save_results:
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(file_dir,'results')

if torch.cuda.is_available():
    device='cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): ## for MAC GPU usage
    device='mps'
else:
    device='cpu'

# To get a distribution of Q from the transition model ensemble, the transition model and Q ensemble must match one-to-one
if alg_name == 'unc_td3':
   q_ensemble_s = fm_ensemble_s
elif alg_name == 'td3':
    q_ensemble_s = 1

setup_logger(seed)

# ------- Setup seeds for reproducibility ------
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# Initialise training loop
train_loop = TrainingLoop(n_total_steps=n_total_steps, n_warmup_steps=n_warmup_steps, exploratory_steps = exploratory_steps, n_dyna_transitions=n_dyna_transitions, env_name=env_name, device=device, buffer_s=buffer_s,batch_s=batch_s,fm_model_batch_s=fm_model_batch_s, fm_n_model_training_iter=fm_n_model_training_iter, fm_model_training_freq=fm_model_training_freq, fm_n_units=fm_n_units, fm_n_layers=fm_n_layers, fm_ensemble_s=fm_ensemble_s, fm_activation=fm_activation, fm_lr=fm_lr, q_n_units=q_n_units, q_n_layers=q_n_layers, q_ensemble_s=q_ensemble_s, q_activation=q_activation, q_lr=q_lr, a_n_units=a_n_units, a_n_layers=a_n_layers, a_activation=a_activation, a_lr=a_lr, alg_name=alg_name, normalize_transitions=normalize_transitions, expl_noise=expl_noise, full_ensemble=full_q_ensemble) 

# train algorithm
logging.info('Training started')
evaluation_acc = train_loop.train()

if save_results:
    if full_q_ensemble:
        ensemb = '_fullEnsemb'
    else:
        ensemb = '_headEnsemb'

    run_dir = os.path.join(file_dir, f'{env_name}',f'{alg_name}',run_type,ensemb,f'{run_number}')
    os.makedirs(run_dir, exist_ok=True)
    torch.save(evaluation_acc, run_dir+'/uncInitPol_evalAcc.pt') 
