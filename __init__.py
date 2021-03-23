import os

import numpy as np
import torch

# to make all paths relative to module folder
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# constants
DTYPE = torch.float64
# convergence seems to be faster with float64, but more stable with float32? more accuracy with float64?
torch.set_default_dtype(DTYPE)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
# torch.set_default_tensor_device(DEVICE) # sadly does not exist

# print more digits, especially for loss
torch.set_printoptions(precision=8, sci_mode=True)

# reproducibility
# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)

os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'
