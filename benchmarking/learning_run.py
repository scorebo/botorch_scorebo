import os
from os.path import dirname, abspath, join
import sys

from functools import partial
import json
from copy import copy

import hydra
import torch
from torch import Tensor
from torch.quasirandom import SobolEngine
import pandas as pd
import numpy as np
        
from ax.service.utils.instantiation import ObjectiveProperties
from ax.service.ax_client import AxClient
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.modelbridge.registry import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from omegaconf import DictConfig
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods.gaussian_likelihood import (
    _GaussianLikelihoodBase,
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from botorch.models import SingleTaskGP
from ax.models.torch.botorch_defaults import _get_acquisition_func
from ax.models.torch.fully_bayesian import (
    single_task_pyro_model,
)
from botorch.models.fully_bayesian import (
    normal_tensor_likelihood,
    normal_tensor_likelihood_noscale,
    SliceFullyBayesianSingleTaskGP,
    SingleTaskPyroModel,
    SingleTaskMeanPyroModel,
    ActiveLearningPyroModel,
    ActiveLearningWithOutputscalePyroModel,
    PyroFullyBayesianSingleTaskGP,
    WideActiveLearningWithOutputscalePyroModel,
    BadActiveLearningWithOutputscalePyroModel
)
from ax.models.torch.fully_bayesian_model_utils import (
    _get_active_learning_gpytorch_model,
    _get_single_task_gpytorch_model,
    slice_bo_prior
)
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.modelbridge.prediction_utils import predict_at_point
from botorch.utils.transforms import unnormalize
from benchmarking.mappings import (
    get_test_function,
    ACQUISITION_FUNCTIONS,
    MODELS,
)


sys.path.append('.')

N_VALID_SAMPLES = int(2e3)
MIN_INFERRED_NOISE_LEVEL = 1e-4


@hydra.main(config_path='./../configs', config_name='conf')
def main(cfg: DictConfig) -> None:
    print(cfg)
    torch.manual_seed(int(cfg.seed))
    if cfg.model.gp == 'Modular':
        acq_func = ACQUISITION_FUNCTIONS[cfg.algorithm.acq_func]

    prior_name = cfg.model.prior_name

    if prior_name == 'BAL':
        model_constructor = _get_active_learning_gpytorch_model
        pyro_model = single_task_pyro_model
        kernel_type = 'rbf'
    elif prior_name == 'BO':
        model_constructor = _get_single_task_gpytorch_model
        pyro_model = single_task_pyro_model
        kernel_type = 'matern'

    # This is what we run for hte acive learning experiments
    elif prior_name == 'AL_slice':
        model_constructor = _get_active_learning_gpytorch_model
        log_likelihood_fun = normal_tensor_likelihood_noscale
        kernel_type = 'rbf'

    # This one is never really run since we can't properly setup acquisition functions
    # Need to run Models.BOTORCH_MODULAR
    elif prior_name == 'BO_slice':
        model_constructor = _get_single_task_gpytorch_model
        log_likelihood_fun = normal_tensor_likelihood
        kernel_type = 'matern'

    if cfg.model.model_kwargs is None:
        model_kwargs = {}
    else:
        model_kwargs = dict(cfg.model.model_kwargs)
    refit_on_update = not hasattr(cfg.model, 'model_parameters')

    refit_params = {}

    if not refit_on_update:
        model_enum = Models.BOTORCH_MODULAR_NOTRANS
        if hasattr(cfg.benchmark, 'model_parameters'):
            params = cfg.benchmark.model_parameters
            refit_params['outputscale'] = Tensor(params.outputscale)
            refit_params['lengthscale'] = Tensor(params.lengthscale)

        # if the model also has fixed parameters, we will override with those
        if hasattr(cfg.model, 'model_parameters'):
            params = cfg.benchmark.model_parameters
            refit_params['outputscale'] = Tensor(params.outputscale)
            refit_params['lengthscale'] = Tensor(params.lengthscale)
    else:
        model_enum = Models.BOTORCH_MODULAR

    acq_name = cfg.algorithm.name
    benchmark = cfg.benchmark.name

    if hasattr(cfg.benchmark, 'outputscale'):
        test_function = get_test_function(
            benchmark, float(cfg.benchmark.noise_std), cfg.seed, float(cfg.benchmark.outputscale))

    else:
        test_function = get_test_function(
            benchmark, float(cfg.benchmark.noise_std), cfg.seed)
    num_init = cfg.benchmark.num_init
    num_bo = cfg.benchmark.num_iters - num_init
    bounds = torch.transpose(torch.Tensor(cfg.benchmark.bounds), 1, 0)
    opt_setup = cfg.acq_opt

    if cfg.model.model_kwargs is None:
        model_kwargs = {}
    else:
        model_kwargs = dict(cfg.model.model_kwargs)

    refit_params = {}
    if hasattr(cfg.algorithm, 'acq_kwargs'):
        acq_func_kwargs = dict(cfg.algorithm.acq_kwargs)
    else:
        acq_func_kwargs = {}

    init_kwargs = {"seed": int(cfg.seed)}

    if cfg.model.gp == 'FullyBayesian':
        model_enum = Models.FULLYBAYESIAN
        model_kwargs = {  # Kwargs to pass to `BoTorchModel.__init__`
            "num_samples": cfg.model.num_samples,
            "warmup_steps": cfg.model.warmup_steps,
            # "num_chains": cfg.model.num_chains,
            "prior_type": prior_name,
            "model_constructor": model_constructor,
            "acqf_constructor": partial(
                _get_acquisition_func,
                acquisition_function_name=cfg.algorithm.acq_func
            ),
            "gp_kernel": kernel_type,
        }
        model_kwargs["pyro_model"] = pyro_model
        bo_step = GenerationStep(
            # model=model_enum,
            model=model_enum,
            # No limit on how many generator runs will be produced
            num_trials=num_bo,
            model_kwargs=model_kwargs,
            model_gen_kwargs={"model_gen_options": {  # Kwargs to pass to `BoTorchModel.gen`
                "optimizer_kwargs": dict(opt_setup)},
            },
        )

    elif cfg.model.gp == 'FullyBayesianSlice':
        model_enum = Models.FULLYBAYESIANSLICE
        model_kwargs = {  # Kwargs to pass to `BoTorchModel.__init__`
            "num_samples": cfg.model.num_samples,
            "warmup_steps": cfg.model.warmup_steps,
            # "num_chains": cfg.model.num_chains,
            "prior_type": prior_name,
            "model_constructor": model_constructor,
            "acqf_constructor": partial(
                _get_acquisition_func,
                acquisition_function_name=cfg.algorithm.acq_func
            ),
            "gp_kernel": kernel_type,
        }
        bo_step = GenerationStep(
            # model=model_enum,
            model=model_enum,
            # No limit on how many generator runs will be produced
            num_trials=num_bo,
            model_kwargs=model_kwargs,
            model_gen_kwargs={"model_gen_options": {  # Kwargs to pass to `BoTorchModel.gen`
                "optimizer_kwargs": dict(opt_setup)},
            },
        )
        model_kwargs["log_likelihood_fun"] = log_likelihood_fun

    else:
        if 'gp_' in cfg.benchmark.name:
            model_enum = Models.BOTORCH_MODULAR_NOTRANS
        else:

            model_enum = Models.BOTORCH_MODULAR
        model_kwargs = {  # Kwargs to pass to `BoTorchModel.__init__`
            "num_samples": int(cfg.model.num_samples),
            "warmup": int(cfg.model.warmup),
            "thinning": int(cfg.model.thinning)
        }
        if cfg.model.prior_name == 'SCoreBO_slice':
            model_class = SliceFullyBayesianSingleTaskGP
            model_kwargs['prior'] = slice_bo_prior(test_function.dim)

        elif cfg.model.prior_name == 'SCoreBO_nuts':
            model_class = PyroFullyBayesianSingleTaskGP
            model_kwargs['pyro_model'] = SingleTaskPyroModel

        elif cfg.model.prior_name == 'SCoreBO_nuts_mean':
            model_class = PyroFullyBayesianSingleTaskGP
            model_kwargs['pyro_model'] = SingleTaskMeanPyroModel

        elif cfg.model.prior_name == 'SCoreBO_al':
            model_class = PyroFullyBayesianSingleTaskGP
            model_kwargs['pyro_model'] = ActiveLearningPyroModel
        elif cfg.model.prior_name == 'SCoreBO_al_bo':
            model_class = PyroFullyBayesianSingleTaskGP
            model_kwargs['pyro_model'] = ActiveLearningWithOutputscalePyroModel
        elif cfg.model.prior_name == 'SCoreBO_al_bo_wide':
            model_class = PyroFullyBayesianSingleTaskGP
            model_kwargs['pyro_model'] = WideActiveLearningWithOutputscalePyroModel
        elif cfg.model.prior_name == 'ScoreBO_al_bo_bad':
            model_class = PyroFullyBayesianSingleTaskGP
            model_kwargs['pyro_model'] = BadActiveLearningWithOutputscalePyroModel
            
        bo_step = GenerationStep(
            # model=model_enum,
            model=model_enum,
            # No limit on how many generator runs will be produced
            num_trials=num_bo,
            model_kwargs={  # Kwargs to pass to `BoTorchModel.__init__`
                "surrogate": Surrogate(
                            botorch_model_class=model_class,
                            model_options=model_kwargs
                ),
                "botorch_acqf_class": acq_func,
                "acquisition_options": {**acq_func_kwargs},
                "refit_on_update": refit_on_update,
                "refit_params": refit_params
            },
            model_gen_kwargs={"model_gen_options": {  # Kwargs to pass to `BoTorchModel.gen`
                "optimizer_kwargs": dict(opt_setup)},
            },
        )
    steps = [
        GenerationStep(
            model=Models.SOBOL,
            num_trials=num_init,
            # Otherwise, it's probably just SOBOL
            model_kwargs=init_kwargs,
        )]
    steps.append(bo_step)

    def evaluate(parameters, seed=None):
            x = torch.tensor(
                [[parameters.get(f"x_{i+1}") for i in range(test_function.dim)]])
            if seed is not None:
                bc_eval = test_function.evaluate_true(x, seed=seed).squeeze().tolist()
            else:
                bc_eval = test_function(x).squeeze().tolist()
            # In our case, standard error is 0, since we are computing a synthetic function.
            return {benchmark: bc_eval}

    if cfg.load_run:
        steps = [steps[-1]]

    gs = GenerationStrategy(
        steps=steps
    )

    # Initialize the client - AxClient offers a convenient API to control the experiment
    ax_client = AxClient(generation_strategy=gs)
    # Setup the experiment
    ax_client.create_experiment(
        name=cfg.experiment_name,
        parameters=[
            {
                "name": f"x_{i+1}",
                "type": "range",
                # It is crucial to use floats for the bounds, i.e., 0.0 rather than 0.
                # Otherwise, the parameter would
                "bounds": bounds[:, i].tolist(),
            }
            for i in range(test_function.dim)
        ],
        objectives={
            benchmark: ObjectiveProperties(minimize=False),
        },
    )

    loglik_arr = [0] * num_init
    joint_loglik_arr = [0] * num_init
    rmse_arr = [0] * num_init
    best_guesses = []
    hyperparameters = {}
    scale_hyperparameters = model_enum != Models.BOTORCH_MODULAR_NOTRANS
        
    for i in range(num_init + num_bo):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(
            trial_index=trial_index, raw_data=evaluate(parameters))
        # Unique sobol draws per seed and iteration, but identical for different acqs
        sobol = SobolEngine(dimension=test_function.dim,
                            scramble=True, seed=i + cfg.seed * 100)
        # crucially, these are already in [0, 1]
        test_samples = sobol.draw(n=N_VALID_SAMPLES)

        # These are for computing the active learning metrics
        current_data = ax_client.get_trials_data_frame()[benchmark].to_numpy()
        if (i >= num_init) and (cfg.model.gp != 'Modular'):
            loglik, joint_loglik, rmse = compute_rmse_and_nmll(
                ax_client, test_samples, test_function, benchmark)
            loglik_arr.append(loglik.item())
            rmse_arr.append(rmse.item())
            model = ax_client._generation_strategy.model.model.model.models[0]

            hps = get_model_hyperparameters(model, current_data, scale_hyperparameters=scale_hyperparameters)
            hyperparameters[f'iter_{i}'] = hps

        elif (i >= num_init):
            model = ax_client._generation_strategy.model.model._surrogate.model
            hps = get_model_hyperparameters(model, current_data, scale_hyperparameters=scale_hyperparameters)
            hyperparameters[f'iter_{i}'] = hps

        if cfg.algorithm.name != 'Dummy' and 'hpo' not in cfg.benchmark.name:
            guess_parameters, guess_trial_index = ax_client.get_best_guess()
            best_guesses.append(guess_parameters)

    for i in range(num_init):
        hyperparameters[f'iter_{i}'] = hyperparameters[f'iter_{num_init}']

        results_df = ax_client.get_trials_data_frame()
        configs = torch.tensor(
            results_df.loc[:, ['x_' in col for col in results_df.columns]].to_numpy())
        
    if 'hpo' not in cfg.benchmark.name:
        results_df['True Eval'] = test_function.evaluate_true(configs)

    else:
        guess_parameters, guess_trial_index = ax_client.get_best_guess()
        guesses_tensor = Tensor([list(guess.values()) for guess in best_guesses])

        rounds = int(cfg.num_infer_rounds)
        if 'hpo' in cfg.benchmark.name:        
            infer_values = [0.0] * rounds    
            for i in range(rounds):
                infer_values[i] = evaluate(guess_parameters, seed=i)[cfg.benchmark.name]


        if (cfg.model.gp != 'Modular') and (cfg.algorithm.name != 'Dummy'):
            results_df['MLL'] = loglik_arr
            results_df['RMSE'] = rmse_arr
        elif (cfg.algorithm.name == 'Dummy') or 'hpo' in cfg.benchmark.name:
            pass
        else:
            guesses_tensor = Tensor([list(guess.values()) for guess in best_guesses])
            results_df['Guess values'] = test_function.evaluate_true(guesses_tensor)
        
        if infer_values is not None:
            results_df['Infer values'] = 0
            results_df['Infer values'][-rounds:] = infer_values
    os.makedirs(cfg.result_path, exist_ok=True)
    with open(f"{cfg.result_path}/{ax_client.experiment.name}_hps.json", "w") as f:
        json.dump(hyperparameters, f, indent=2)
    results_df.to_csv(f"{cfg.result_path}/{ax_client.experiment.name}.csv")


def get_model_hyperparameters(model, current_data, scale_hyperparameters=True):
    from gpytorch.kernels import ScaleKernel
    from gpytorch.means import ConstantMean
    
    has_outputscale = isinstance(model.covar_module, ScaleKernel)
    has_mean = isinstance(model.mean_module, ConstantMean)
    def tolist(l): return l.detach().to(torch.float32).numpy().tolist()
    hp_dict = {}

    data_mean = current_data.mean()
    if scale_hyperparameters:
        data_variance = current_data.var()
    else:
        data_variance = torch.Tensor([1])
    
    if has_outputscale:
        hp_dict['outputscale'] = tolist(model.covar_module.outputscale * data_variance)
        hp_dict['lengthscales'] = tolist(model.covar_module.base_kernel.lengthscale)
        hp_dict['noise'] = tolist(model.likelihood.noise * data_variance)
    else:
        hp_dict['lengthscales'] = tolist(model.covar_module.lengthscale)
        hp_dict['noise'] = tolist(model.likelihood.noise)
    if has_mean:
        hp_dict['mean'] = tolist(model.mean_module.constant * data_variance ** 0.5 + data_mean)
    return hp_dict


def compute_rmse_and_nmll(ax_client, test_samples, objective, objective_name):
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.distributions import MultivariateNormal

    output = -objective.evaluate_true(unnormalize(test_samples, objective.bounds))
    y_transform = ax_client._generation_strategy.model.transforms['StandardizeY']
    y_mean, y_std = y_transform.Ymean[objective_name], y_transform.Ystd[objective_name]

    mu, cov = ax_client._generation_strategy.model.model.predict(test_samples)
    mu_true = (mu * y_std + y_mean).flatten()
    
    model = ax_client._generation_strategy.model.model.model.models[0]
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    preds = model(test_samples)
    norm_yvalid = (output - y_mean) / y_std
    predmean = preds.mean.mean(dim=0)
    predcov = preds.variance.mean(dim=0)

    norm_yvalid = norm_yvalid.flatten()
    marg_dist = MultivariateNormal(predmean, torch.diag(predcov))
    loglik = -mll(marg_dist, norm_yvalid).mean()
    rmse = torch.pow(output - mu_true, 2).mean()

    return loglik, torch.Tensor([0]), rmse


if __name__ == '__main__':
    main()
