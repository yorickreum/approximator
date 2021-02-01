import os

import numpy as np
import torch

# to make all paths relative to module folder
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# constants
DTYPE = torch.float64  # convergence seems to be faster with float64
# LOGSTEPS = 1

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# print more digits, especially for loss
torch.set_printoptions(precision=5, sci_mode=True)

# reproducibility
# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)

os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'
