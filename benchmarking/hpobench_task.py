# warnings.filterwarnings('ignore')
from ConfigSpace import Configuration
import sys
import os
from os.path import join, dirname, abspath
from botorch.test_functions.base import BaseTestProblem

try:
    from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark as Benchmark
except:
    pass
    
import numpy as np
import torch
from torch import Tensor


def format_arguments(config_space, args):
    hparams = config_space.get_hyperparameters_dict()
    formatted_hparams = {}
    for param, param_values in hparams.items():
        formatted_hparams[param] = param_values.sequence[args[param]]
    return formatted_hparams


def rescale_arguments(args):
    width, batch_size, alpha, learning_rate_init = args

    # 3 values, nearest alternative
    #depth = np.round((depth + 1e-10) * 2.999999999 + 0.5)
    width = np.round(np.power(2, 4 + (width * 6)))
    batch_size = np.round(np.power(2, 2 + (batch_size * 6)))
    alpha = np.power(10, -8 + (alpha * 5))
    learning_rate_init = np.power(10, -5 + (learning_rate_init * 5))
    config = {
        'depth': 2,
        'width': int(width),
        'batch_size': int(batch_size),
        'alpha': alpha,
        'learning_rate_init': learning_rate_init
    }
    print(config)
    return config


class HPOBenchFunction(BaseTestProblem):

    def __init__(self, task_id: int, noise_std: float = None, negate: bool = False, seed: int = 42):
        self.task_id = task_id
        self.seed = seed
        self.dim = 4
        self._bounds = [(0.0, 1.0) * self.dim]
        super().__init__(noise_std=noise_std, negate=negate)
        
    def evaluate_true(self, X: Tensor, seed=None) -> Tensor:
        b = Benchmark(task_id=self.task_id, rng=self.seed)
        config_space = b.get_configuration_space(seed=self.seed)
        
        args_dict = rescale_arguments(tuple(*X.detach().numpy()))
        config = Configuration(b.get_configuration_space(seed=self.seed), args_dict)
        if seed is not None:
            print(seed)
            result_dict = b.objective_function(configuration=config, rng=seed)
        else:
            result_dict = b.objective_function(configuration=config, rng=self.seed)
        val = result_dict['function_value']

        return Tensor([val])
