from os.path import dirname, join, abspath
from glob import glob
import json

import numpy as np
from scipy.optimize import minimize
import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from benchmarking.gp_sample.gp_task import GPTestFunction
from botorch.optim import optimize_acqf
gps = glob(join(dirname(abspath(__file__)), 'gp_*dim/*.csv'))

for gp in gps:
    seed = int(gp.split('_run')[-1].split('.')[0])
    dim = int(gp.split('gp_')[-1].split('dim')[0])

    sobol = SobolEngine(dimension=dim)
    fun = GPTestFunction(dim=dim, noise_std=0, seed=seed, negate=True)

    x, opt = optimize_acqf(fun, fun.bounds, q=1, num_restarts=100,
                           raw_samples=65532, options=dict(maxiter=200))
    x = x.detach().numpy().tolist()
    opt = opt.item()

    d = {'opt': opt, 'X': x}
    fp = join(dirname(abspath(__file__)), f'gp_{dim}dim/gp_{dim}dim_run{seed}.json')
    with open(fp, 'w') as j:
        json.dump(d, j)
