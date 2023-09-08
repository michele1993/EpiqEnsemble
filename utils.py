import torch
import logging
import gym


def to_torch(x):

    x = torch.from_numpy(x).float()
    if x.ndimension() == 1:
       x = x.unsqueeze(0)
    return x

def to_np(x):
    x = x.detach().cpu().numpy()
    if len(x.shape) >= 1:
       x = x.squeeze(0)
    return x    

def setup_logger(seed):

    logging.basicConfig(format='%(asctime)s %(message)s', encoding='utf-8', level=logging.INFO)

    logging.debug(f'Pytorch version: {torch.__version__}')
    logging.debug(f'Gym version: {gym.__version__}')
    if seed is not None:
        logging.info(f'Seed: {seed}')

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


