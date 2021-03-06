import os
import numpy as np  # https://stackoverflow.com/questions/11788950/importing-numpy-into-functions
import torch
import matplotlib.pyplot as plt
import time

randseed = int(time.time()) #123 # set random seed for reproducible results
np.random.seed(randseed)
torch.manual_seed(randseed)

dir = r'/Users/andrewzahorodnii/rnn/'  # use r'path with spaces' if there are spaces in the path name
device = (torch.device("cuda:0") if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
#TODO: figure out where to put .to(DEVICE)