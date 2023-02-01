#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Active learning acquisition functions.

.. [Seo2014activedata]
    S. Seo, M. Wallat, T. Graepel, and K. Obermayer. Gaussian process regression:
    Active data selection and test point rejection. IJCNN 2000.

.. [Chen2014seqexpdesign]
    X. Chen and Q. Zhou. Sequential experimental designs for stochastic kriging.
    Winter Simulation Conference 2014.

.. [Binois2017repexp]
    M. Binois, J. Huang, R. B. Gramacy, and M. Ludkovski. Replication or
    exploration? Sequential design for stochastic simulation experiments.
    ArXiv 2017.
"""

from __future__ import annotations

from typing import Optional

from math import pi
from copy import deepcopy

import torch
from botorch import settings
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.models.model import Model
from botorch.models.utils import check_no_nans
from botorch.models.utils import fantasize as fantasize_flag
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from botorch.acquisition import AcquisitionFunction
from torch import Tensor
from gpytorch.mlls import ExactMarginalLogLikelihood

# FOR PLOTTING
import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt


MCMC_DIM = -3  # Location of the MCMC batch dimension
CLAMP_LB = 1e-6


class qNegIntegratedPosteriorVariance(AnalyticAcquisitionFunction):
    r"""Batch Integrated Negative Posterior Variance for Active Learning.

    This acquisition function quantifies the (negative) integrated posterior variance
    (excluding observation noise, computed using MC integration) of the model.
    In that, it is a proxy for global model uncertainty, and thus purely focused on
    "exploration", rather the "exploitation" of many of the classic Bayesian
    Optimization acquisition functions.

    See [Seo2014activedata]_, [Chen2014seqexpdesign]_, and [Binois2017repexp]_.
    """

    def __init__(
        self,
        model: Model,
        mc_points: Tensor,
        sampler: Optional[MCSampler] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        **kwargs,
    ) -> None:
        r"""q-Integrated Negative Posterior Variance.

        Args:
            model: A fitted model.
            mc_points: A `batch_shape x N x d` tensor of points to use for
                MC-integrating the posterior variance. Usually, these are qMC
                samples on the whole design space, but biased sampling directly
                allows weighted integration of the posterior variance.
            sampler: The sampler used for drawing fantasy samples. In the basic setting
                of a standard GP (default) this is a dummy, since the variance of the
                model after conditioning does not actually depend on the sampled values.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            X_pending: A `n' x d`-dim Tensor of `n'` design points that have
                points that have been submitted for function evaluation but
                have not yet been evaluated.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        if sampler is None:
            # If no sampler is provided, we use the following dummy sampler for the
            # fantasize() method in forward. IMPORTANT: This assumes that the posterior
            # variance does not depend on the samples y (only on x), which is true for
            # standard GP models, but not in general (e.g. for other likelihoods or
            # heteroskedastic GPs using a separate noise model fit on data).
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1]))
        self.sampler = sampler
        self.X_pending = X_pending
        self.register_buffer("mc_points", mc_points)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        # Construct the fantasy model (we actually do not use the full model,
        # this is just a convenient way of computing fast posterior covariances
        fantasy_model = self.model.fantasize(
            X=X, sampler=self.sampler, observation_noise=True
        )

        bdims = tuple(1 for _ in X.shape[:-2])
        if self.model.num_outputs > 1:
            # We use q=1 here b/c ScalarizedObjective currently does not fully exploit
            # LinearOperator operations and thus may be slow / overly memory-hungry.
            # TODO (T52818288): Properly use LinearOperators in scalarize_posterior
            mc_points = self.mc_points.view(-1, *bdims, 1, X.size(-1))
        else:
            # While we only need marginal variances, we can evaluate for q>1
            # b/c for GPyTorch models lazy evaluation can make this quite a bit
            # faster than evaluting in t-batch mode with q-batch size of 1
            mc_points = self.mc_points.view(*bdims, -1, X.size(-1))

        # evaluate the posterior at the grid points
        with settings.propagate_grads(True):
            posterior = fantasy_model.posterior(
                mc_points, posterior_transform=self.posterior_transform
            )

        neg_variance = posterior.variance.mul(-1.0)

        if self.posterior_transform is None:
            # if single-output, shape is 1 x batch_shape x num_grid_points x 1
            return neg_variance.mean(dim=-2).squeeze(-1).squeeze(0)
        else:
            # if multi-output + obj, shape is num_grid_points x batch_shape x 1 x 1
            return neg_variance.mean(dim=0).squeeze(-1).squeeze(-1)


class PairwiseMCPosteriorVariance(MCAcquisitionFunction):
    r"""Variance of difference for Active Learning

    Given a model and an objective, calculate the posterior sample variance
    of the objective on the difference of pairs of points. See more implementation
    details in `forward`. This acquisition function is typically used with a
    pairwise model (e.g., PairwiseGP) and a likelihood/link function
    on the pair difference (e.g., logistic or probit) for pure exploration
    """

    def __init__(
        self,
        model: Model,
        objective: MCAcquisitionObjective,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        r"""Pairwise Monte Carlo Posterior Variance

        Args:
            model: A fitted model.
            objective: An MCAcquisitionObjective representing the link function
                (e.g., logistic or probit.) applied on the difference of (usually 1-d)
                two samples. Can be implemented via GenericMCObjective.
            sampler: The sampler used for drawing MC samples.
        """
        super().__init__(
            model=model, sampler=sampler, objective=objective, X_pending=None
        )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate PairwiseMCPosteriorVariance on the candidate set `X`.

        Args:
            X: A `batch_size x q x d`-dim Tensor. q should be a multiple of 2.

        Returns:
            Tensor of shape `batch_size x q` representing the posterior variance
            of link function at X that active learning hopes to maximize
        """
        if X.shape[-2] == 0 or X.shape[-2] % 2 != 0:
            raise RuntimeError(
                "q must be a multiple of 2 for PairwiseMCPosteriorVariance"
            )

        # The output is of shape batch_shape x 2 x d
        # For PairwiseGP, d = 1
        post = self.model.posterior(X)
        samples = self.get_posterior_samples(post)  # num_samples x batch_shape x 2 x d

        # The output is of shape num_samples x batch_shape x q/2 x d
        # assuming the comparison is made between the 2 * i and 2 * i + 1 elements
        samples_diff = samples[..., ::2, :] - samples[..., 1::2, :]
        mc_var = self.objective(samples_diff).var(dim=0)
        mean_mc_var = mc_var.mean(dim=-1)

        return mean_mc_var


class BALM(AcquisitionFunction):
    def __init__(
            self,
            model: Model,
            **kwargs: Any
    ) -> None:

        super().__init__(model)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X, observation_noise=True)
        return posterior.variance.mean(dim=MCMC_DIM).squeeze(-1)


class BALD(AcquisitionFunction):
    def __init__(
        self,
        model: Model,
        **kwargs: Any
    ) -> None:

        super().__init__(model)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X, observation_noise=True)
        marg_variance = posterior.variance.mean(dim=MCMC_DIM)
        cond_variances = posterior.variance
        marg_entropy = compute_gaussian_entropy(marg_variance)
        cond_entropy = compute_gaussian_entropy(cond_variances)
        bald = marg_entropy - cond_entropy.mean(dim=MCMC_DIM)

        return bald.squeeze(-1)

class QBMGP(AcquisitionFunction):
    def __init__(
        self,
        model: Model,
        **kwargs: Any
    ) -> None:

        super().__init__(model)


    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        if X.ndim == 3:
            return_full = True
            X = X.unsqueeze(1)
        else:
            return_full = False

        posterior = self.model.posterior(X, observation_noise=True)
        posterior_mean = posterior.mean
        marg_mean = posterior_mean.mean(dim=MCMC_DIM).unsqueeze(-1)
        var_of_mean = torch.pow(
            marg_mean - posterior_mean, 2).mean(dim=MCMC_DIM)
        mean_of_var = posterior.variance.mean(dim=MCMC_DIM)
        return (var_of_mean + mean_of_var).squeeze(-1)


class BQBC(AcquisitionFunction):
    def __init__(
        self,
        model: Model,
        **kwargs: Any
    ) -> None:

        super().__init__(model)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X, observation_noise=True)
        posterior_mean = posterior.mean
        marg_mean = posterior_mean.mean(dim=MCMC_DIM)
        var_of_mean = torch.pow(marg_mean - posterior_mean.squeeze(-1), 2)
        return var_of_mean.squeeze(-1)

class StatisticalDistance(AcquisitionFunction):
    """Statistical distance-based Bayesian Active Learning

    Args:
        AcquisitionFunction ([type]): [description]
    """

    def __init__(
        self,
        model: Model,
        noisy_distance: bool = True,
        distance_metric: str = 'HR',
        **kwargs: Any
    ) -> None:

        super().__init__(model)
        self.noisy_distance = noisy_distance
        available_dists = ['JS', 'KL', 'WS', 'BC', 'HR']
        if distance_metric not in available_dists:
            raise ValueError(f'Distance metric {distance_metric}'
            f'Not available. Choose any of {available_dists}')
        self.distance = DISTANCE_METRICS[distance_metric]

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        if X.ndim == 3:
            return_full = True
            X = X.unsqueeze(1)
        else:
            return_full = False

        posterior = self.model.posterior(X, observation_noise=self.noisy_distance)
        cond_means = posterior.mean
        marg_mean = cond_means.mean(MCMC_DIM).unsqueeze(-1)
        cond_variances = posterior.variance
        marg_variance = cond_variances.mean(MCMC_DIM).unsqueeze(-1)

        dist_al = self.distance(cond_means, marg_mean, cond_variances, marg_variance)
            
        if return_full:
            return dist_al.squeeze(-1).squeeze(-1)
        return dist_al.mean(MCMC_DIM).squeeze(-1)


class MaxValueSelfCorrecting(AcquisitionFunction):
    """Wasserstein-metric based Bayesian Active Learning

    Args:
        AcquisitionFunction ([type]): [description]
    """

    def __init__(
        self,
        model: Model,
        optimal_inputs: Tensor,
        optimal_outputs: Tensor,
        paths,
        noisy_distance: bool = True,
        split_coefficient: float = -1.0,
        distance_metric: str = 'WS',
        **kwargs: Any
    ) -> None:
        super().__init__(model)
        self.noisy_distance = noisy_distance
        self.optimal_inputs = optimal_inputs
        self.optimal_outputs = optimal_outputs
        self.num_samples = len(optimal_outputs)
        self.num_models = self.optimal_outputs.shape[1]
        self.split_coefficient = split_coefficient

        if distance_metric not in DISTANCE_METRICS.keys():
            raise ValueError(f'Distance metric {distance_metric}'
            f'Not available. Choose any of {available_dists}')
        self.distance = DISTANCE_METRICS[distance_metric]
        
        # pretty much only used for plotting
        self.noisy_distance = noisy_distance

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        prev_m = self.model.posterior(
            X.unsqueeze(-3), observation_noise=self.noisy_distance
        )
        noiseless_var = self.model.posterior(
            X.unsqueeze(-3), observation_noise=False
        ).variance.repeat(1, 1, 1, self.num_samples)
        # Compute the mixture mean and variance
        cond_means = prev_m.mean.repeat(1, 1, 1, self.num_samples)
        cond_variances = prev_m.variance.repeat(1, 1, 1, self.num_samples)
        marg_mean = cond_means.mean(MCMC_DIM).unsqueeze(
            1).repeat(1, self.num_models, 1, 1)
        marg_variance = cond_variances.mean(MCMC_DIM).unsqueeze(
            1).repeat(1, self.num_models, 1, 1)

        mean_m = cond_means
        # batch_shape x 1
        variance_m = cond_variances.clamp_min(CLAMP_LB)
        # batch_shape x 1
        check_no_nans(variance_m)
        # get stdv of noiseless variance
        stdv = noiseless_var.sqrt()
        # batch_shape x 1
        normal = torch.distributions.Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )

        # prepare max value quantities required by GIBBON
        mvs = self.optimal_outputs.permute(
            1, 2, 0).unsqueeze(0).repeat(X.shape[0], 1, 1, 1)
        normalized_mvs = (mvs - mean_m) / stdv
        cdf_mvs = normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)
        pdf_mvs = torch.exp(normal.log_prob(normalized_mvs))

        mean_truncated = mean_m - stdv * pdf_mvs / cdf_mvs

        # This is the noiseless variance (i.e. the part that gets truncated)
        var_truncated = noiseless_var * \
            (1 - normalized_mvs * pdf_mvs / cdf_mvs - torch.pow(pdf_mvs / cdf_mvs, 2))

        var_truncated = var_truncated + (variance_m - noiseless_var)
        marg_truncated_mean = mean_truncated.mean(MCMC_DIM, keepdim=True).mean(-1, keepdim=True)
        marg_truncated_var = var_truncated.mean(MCMC_DIM, keepdim=True).mean(-1, keepdim=True)
        if self.split_coefficient < 0:
            dist = self.distance(mean_truncated, marg_truncated_mean, var_truncated, marg_truncated_var)
            
        else:
            dist_bo = self.distance(mean_truncated, marg_truncated_mean, var_truncated, marg_truncated_var)
            dist_al = self.distance(cond_means, marg_mean, cond_variances, marg_variance)
            dist = dist_al + dist_bo * self.split_coefficient
        return dist.mean(-1).squeeze(-1)


class JointSelfCorrecting(AcquisitionFunction):
    """Not really the correct name since it is not actually entropy search,
    but it's quite nice anyway.

    Args:
        AcquisitionFunction ([type]): [description]
    """

    def __init__(
        self,
        model: Model,
        optimal_inputs: Tensor,
        optimal_outputs: Tensor,
        noisy_distance: bool = True,
        condition_noiseless: bool = True,
        split_coefficient: float = -1.0,
        paths=None,
        distance_metric: str = 'HR',
        **kwargs: Any
    ) -> None:
        super().__init__(model)
        self.condition_noiseless = condition_noiseless
        self.initial_model = model
        self.optimal_inputs = optimal_inputs.permute(1, 0, 2)
        self.optimal_outputs = optimal_outputs.permute(1, 0, 2)
        self.num_samples = len(optimal_outputs)
        self.noisy_distance = noisy_distance
        self.num_models = optimal_inputs.shape[1]
        self.split_coefficient = split_coefficient

        available_dists = ['JS', 'KL', 'WS', 'BC', 'HR']
        if distance_metric not in available_dists:
            raise ValueError(f'Distance metric {distance_metric}'
            f'Not available. Choose any of {available_dists}')
        self.distance = DISTANCE_METRICS[distance_metric]
        
        from botorch.models.model_list_gp_regression import ModelListGP
        with fantasize_flag():
            with settings.propagate_grads(False):
                post_ps = self.initial_model.posterior(
                    self.model.train_inputs[0], observation_noise=False
                )

            if self.condition_noiseless:
                conditional_model_list = [self.initial_model.condition_on_observations(
                    X=self.initial_model.transform_inputs(
                        self.optimal_inputs[:, sample_idx]).unsqueeze(1),
                    Y=self.optimal_outputs[:, sample_idx].unsqueeze(1),
                    # <--- THIS IS THE ONLY DIFFERENCE
                    noise=CLAMP_LB * \
                    torch.ones_like(self.optimal_outputs[:, sample_idx]).unsqueeze(1),
                ) for sample_idx in range(self.num_samples)]
                self.conditional_model = ModelListGP(*conditional_model_list)
                # Condition without observation noise.
            else:
                conditional_model_list = [self.initial_model.condition_on_observations(
                    X=self.initial_model.transform_inputs(
                        self.optimal_inputs[:, sample_idx]).unsqueeze(1),
                    Y=self.optimal_outputs[:, sample_idx].unsqueeze(1),
                    # <--- THIS IS THE ONLY DIFFERENCE
                ) for sample_idx in range(self.num_samples)]
                self.conditional_model = ModelListGP(*conditional_model_list)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        prev_m = self.initial_model.posterior(
            X.unsqueeze(-3), observation_noise=self.noisy_distance)
        # Compute the mixture mean and variance
        marg_mean = prev_m.mean.mean(MCMC_DIM).unsqueeze(
            1).repeat(1, self.num_models, 1, self.num_samples)
        marg_variance = prev_m.variance.mean(MCMC_DIM).unsqueeze(
            1).repeat(1, self.num_models, 1, self.num_samples)

        posterior_m = self.conditional_model.posterior(
            X.unsqueeze(-3), observation_noise=self.noisy_distance)
        noiseless_var = self.conditional_model.posterior(
            X.unsqueeze(-3), observation_noise=False
        ).variance

        cond_means = posterior_m.mean
        cond_variances = posterior_m.variance

        mean_m = cond_means
        # batch_shape x 1
        variance_m = cond_variances.clamp_min(CLAMP_LB)
        # batch_shape x 1
        check_no_nans(variance_m)
        # get stdv of noiseless variance
        stdv = noiseless_var.sqrt()
        # batch_shape x 1
        normal = torch.distributions.Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )

        # prepare max value quantities required by GIBBON
        mvs = self.optimal_outputs.transpose(
            2, 1).unsqueeze(0).repeat(X.shape[0], 1, 1, 1)
        normalized_mvs = (mvs - mean_m) / stdv
        cdf_mvs = normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)
        pdf_mvs = torch.exp(normal.log_prob(normalized_mvs))

        mean_truncated = mean_m - stdv * pdf_mvs / cdf_mvs

        # This is the noiseless variance (i.e. the part that gets truncated)
        var_truncated = noiseless_var * \
            (1 - normalized_mvs * pdf_mvs / cdf_mvs - torch.pow(pdf_mvs / cdf_mvs, 2))

        var_truncated = var_truncated + (variance_m - noiseless_var)
        # The last dim is the num_samples dim
        marg_truncated_mean = mean_truncated.mean(MCMC_DIM, keepdim=True).mean(-1, keepdim=True)
        marg_truncated_var = var_truncated.mean(MCMC_DIM, keepdim=True).mean(-1, keepdim=True)
        if self.split_coefficient < 0:
            dist = self.distance(mean_truncated, marg_truncated_mean, var_truncated, marg_truncated_var)
            
        elif self.split_coefficient == 0:
            dist_bo = self.distance(mean_truncated, marg_truncated_mean, var_truncated, marg_truncated_var)
            dist_al = self.distance(mean_truncated, prev_m.mean, var_truncated, prev_m.variance)

            dist = torch.log(dist_bo) + torch.log(dist_al)

        else:
            dist_bo = self.distance(mean_truncated, marg_truncated_mean, var_truncated, marg_truncated_var)
            dist_al = self.distance(mean_truncated, prev_m.mean, var_truncated, prev_m.variance)

            dist = dist_bo * dist_al
            
            
        return dist.mean(-1).squeeze(-1)


class Dummy(AcquisitionFunction):
    """Wasserstein-metric based Bayesian Active Learning

    Args:
        AcquisitionFunction ([type]): [description]
    """

    def __init__(
        self,
        model: Model,
        **kwargs: Any
    ) -> None:

        super().__init__(model)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        return (X.sum(axis=-1)).flatten().unsqueeze(-1)


def wasserstein_distance(mean_x, mean_y, var_x, var_y):
    mean_term = torch.pow(mean_x - mean_y, 2) 
    var_term = var_x + var_y - 2 * torch.sqrt(var_x * var_y)

    return torch.sqrt(mean_term + var_term)


def bhattacharya_distance(mean_x, mean_y, var_x, var_y):
    mean_term = 0.25 * torch.pow(mean_x - mean_y, 2) / (var_x + var_y)
    var_term =  0.5 * torch.log((var_x + var_y) / torch.sqrt(2 * var_x * var_y))
    return mean_term + var_term        
            
            
def hellinger_distance(mean_x, mean_y, var_x, var_y):
    exp_term = -0.25 * torch.pow(mean_x - mean_y, 2) / (var_x + var_y)
    mult_term =  torch.sqrt(torch.sqrt(2 * var_x * var_y) / (var_x + var_y))
    return torch.sqrt(1 - mult_term * torch.exp(exp_term))      
            

def symmetric_kl_divergence(mean_x, mean_y, var_x, var_y):
    kl_first_term = torch.log(torch.sqrt(var_x / var_y)) + torch.log(torch.sqrt(var_y / var_x))
    second_term_mean_x = 0.5 * (torch.pow(mean_x - mean_y, 2) + var_x) / var_y
    second_term_mean_y = 0.5 * (torch.pow(mean_y - mean_x, 2) + var_y) / var_x
    kl = 0.5 * (kl_first_term + second_term_mean_x + second_term_mean_y)
    return kl


def jensen_shannon_divergence(mean_x, mean_y, var_x, var_y):
    w_dist_mean = (mean_x + mean_y) / 2
    w_dist_var = (var_x + var_y) / 2
    js_first_term = torch.log(torch.sqrt(var_x / w_dist_var)) + torch.log(torch.sqrt(var_y / w_dist_var))
    second_term_mean_x = 0.5 * (torch.pow(mean_x - w_dist_mean, 2) + var_x) / w_dist_var
    second_term_mean_y = 0.5 * (torch.pow(mean_y - w_dist_mean, 2) + var_y) / w_dist_var
    js = 0.5 * (js_first_term + second_term_mean_x + second_term_mean_y)
    return js


DISTANCE_METRICS = {
    'WS': wasserstein_distance,
    'BC': bhattacharya_distance,
    'HR': hellinger_distance,
    'KL': symmetric_kl_divergence,
    'JS': jensen_shannon_divergence,
}
