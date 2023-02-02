from botorch.models.gp_regression import SingleTaskGP, FixedNoiseGP
from botorch.test_functions import (
    Branin,
    Hartmann,
    Rosenbrock,
    Forrester
)

from ax.modelbridge.registry import Models
from benchmarking.gp_sample.gp_task import GPTestFunction
from benchmarking.synthetic import (
    Gramacy1, 
    Gramacy2, 
    Higdon, 
    Ishigami, 
)
from benchmarking.hpobench_task import HPOBenchFunction

from botorch.acquisition import (
    qNoisyExpectedImprovement,
)
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
)

from botorch.acquisition.joint_entropy_search import (
    qLowerBoundJointEntropySearch,
    qExploitLowerBoundJointEntropySearch,
)
from botorch.acquisition.active_learning import (
    JointSelfCorrecting,
    MaxValueSelfCorrecting,
    StatisticalDistance,
    QBMGP,
    BALM,
    BQBC
)
import torch

def get_test_function(name: str, noise_std: float, seed: int = 0, outputscale=1):

    TEST_FUNCTIONS = {
        'gramacy1': Gramacy1(noise_std=noise_std, negate=True),
        'higdon': Higdon(noise_std=noise_std, negate=True),
        'gramacy2': Gramacy2(noise_std=noise_std, negate=True),
        'ishigami': Ishigami(noise_std=noise_std, negate=True),
        'active_branin': Branin(noise_std=noise_std, negate=True),
        'active_hartmann6': Hartmann(dim=6, noise_std=noise_std, negate=True),
        'branin': Branin(noise_std=noise_std, negate=True),
        'hartmann3': Hartmann(dim=3, noise_std=noise_std, negate=True),
        'hartmann4': Hartmann(dim=4, noise_std=noise_std, negate=True),
        'hartmann6': Hartmann(dim=6, noise_std=noise_std, negate=True),
        'rosenbrock2': Rosenbrock(dim=2, noise_std=noise_std, negate=True),
        'rosenbrock4': Rosenbrock(dim=4, noise_std=noise_std, negate=True),
        'forrester': Forrester(dim=1, noise_std=noise_std, negate=True),
        'hpo_segment': HPOBenchFunction(negate=True, seed=seed, task_id=146822),
        'hpo_blood': HPOBenchFunction(negate=True, seed=seed, task_id=10101),
    }
    if 'gp' in name:
        TEST_FUNCTIONS = {
            'gp_4dim': GPTestFunction(dim=4, noise_std=noise_std, seed=seed, negate=True),
    
        }
    test_function = TEST_FUNCTIONS[name]
    return test_function


ACQUISITION_FUNCTIONS = {
    'NEI': qNoisyExpectedImprovement,
    'JES': qLowerBoundJointEntropySearch,
    'JES-e': qExploitLowerBoundJointEntropySearch,
    'GIBBON': qLowerBoundMaxValueEntropy,
    'ScoreBO_M': MaxValueSelfCorrecting,
    'ScoreBO_J': JointSelfCorrecting,
    'SAL': StatisticalDistance,
    'QBMGP': QBMGP,
    'BALM': BALM,
    'BQBC': BQBC
}

MODELS = {
    'FixedNoiseGP': FixedNoiseGP,
    'SingleTaskGP': SingleTaskGP,
}


INITS = {
    'sobol': Models.SOBOL,
}
