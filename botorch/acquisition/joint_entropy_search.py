#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition function for joint entropy search (JES). The code utilizes the
implementation designed for the multi-objective batch setting.

"""

from __future__ import annotations

from typing import Any, Optional
from math import pi
import torch.distributions as dist

import torch
from torch import Tensor

from torch.distributions import Normal
from botorch.acquisition.multi_objective.joint_entropy_search import (
    qLowerBoundMultiObjectiveJointEntropySearch,
)
from botorch import settings
from botorch.models.utils import fantasize as fantasize_flag
from botorch.acquisition.multi_objective.utils import compute_sample_box_decomposition
from botorch.models.model import Model
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.gp_regression import SingleTaskGP, FixedNoiseGP
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import is_fully_bayesian

NEG_INF = -1e10
CLAMP_LB = 1e-6


class qLowerBoundJointEntropySearch(AcquisitionFunction, MCSamplerMixin):
    r"""The acquisition function for the Joint Entropy Search, where the batches
    `q > 1` are supported through the lower bound formulation.

    This acquisition function computes the mutual information between the observation
    at a candidate point `X` and the optimal input-output pair.

    See [Tu2022]_ for a discussion on the estimation procedure.
    """

    def __init__(
        self,
        model: Model,
        optimal_inputs: Tensor,
        optimal_outputs: Tensor,
        condition_noiseless: bool = True,
        maximize: bool = True,
        hypercell_bounds: Tensor = None,
        X_pending: Optional[Tensor] = None,
        estimation_type: str = "LB2",
        num_samples: int = 64,
        **kwargs: Any,
    ) -> None:
        r"""Joint entropy search acquisition function.

        Args:
            model: A fitted single-outcome model.
            optimal_inputs: A `num_samples x d`-dim tensor containing the sampled
                optimal inputs of dimension `d`. We assume for simplicity that each
                sample only contains one optimal set of inputs.
            optimal_outputs: A `num_samples x 1`-dim Tensor containing the optimal
                set of objectives of dimension `1`.
            maximize: If true, we consider a maximization problem.
            hypercell_bounds:  A `num_samples x 2 x J x 1`-dim Tensor containing the
                hyper-rectangle bounds for integration, where `J` is the number of
                hyper-rectangles. By default, the problem is assumed to be
                unconstrained and therefore the region of integration for a sample
                `(x*, y*)` is a `J=1` hyper-rectangle of the form  `(-infty, y^*]`
                for a maximization problem and `[y^*, +infty)` for a minimization
                problem. In the constrained setting, the region of integration also
                includes the infeasible space.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation, but have not yet been evaluated.
            estimation_type: A string to determine which entropy estimate is
                computed: "0", "LB", "LB2", or "MC". In the single-objective
                setting, "LB" is equivalent to "LB2".
            num_samples: The number of Monte Carlo samples used for the Monte Carlo
                estimate.
        """

        super().__init__(model=model)
        
        self.batch_dim = optimal_inputs.shape[1]
        self.optimal_inputs = optimal_inputs.permute(1, 0, 2)
        self.optimal_outputs = optimal_outputs.permute(1, 0, 2)

        self.num_samples = optimal_inputs.shape[0]
        self.num_models = optimal_inputs.shape[1]
        self.condition_noiseless = condition_noiseless
        self.initial_model = model

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

        self.estimation_type = estimation_type
        self.set_X_pending(X_pending)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluates qLowerBoundJointEntropySearch at the design points `X`.
        Args:
            X: A `batch_shape x q x d`-dim Tensor of `batch_shape` t-batches with `q`
            `d`-dim design points each.
        Returns:
            A `batch_shape`-dim Tensor of acquisition values at the given design
            points `X`.
        """

        return self._compute_lower_bound_information_gain(X)

    def _compute_posterior_statistics(
        self, X: Tensor
    ) -> dict[str, Union[Tensor, GPyTorchPosterior]]:
        r"""Compute the posterior statistics.
        Args:
            X: A `batch_shape x q x d`-dim Tensor of inputs.

        Returns:
            A dictionary containing the posterior variables used to estimate the
            entropy.

            - "initial_entropy": A `batch_shape`-dim Tensor containing the entropy of
                the Gaussian random variable `p(Y| X, D_n)`.
            - "posterior_mean": A `batch_shape x num_pareto_samples x q x 1 x M`-dim
                Tensor containing the posterior mean at the input `X`.
            - "posterior_variance": A `batch_shape x num_pareto_samples x q x 1 x M`
                -dim Tensor containing the posterior variance at the input `X`
                excluding the observation noise.
            - "observation_noise": A `batch_shape x num_pareto_samples x q x 1 x M`
                -dim Tensor containing the observation noise at the input `X`.
            - "posterior_with_noise": The posterior distribution at `X` which
                includes the observation noise. This is used to compute the marginal
                log-probabilities with respect to `p(y| x, D_n)` for `x` in `X`.
        """
        tkwargs = {"dtype": X.dtype, "device": X.device}
        CLAMP_LB = torch.finfo(tkwargs["dtype"]).eps
        X = X.squeeze(1)
        initial_posterior_plus_noise = self.initial_model.posterior(
            X, observation_noise=True
        )
        # Additional constant term.
        add_term = (
            0.5
            * self.model.num_outputs
            * (1 + torch.log(2 * pi * torch.ones(1, **tkwargs)))
        )

        initial_entropy = add_term + 0.5 * torch.log(
            initial_posterior_plus_noise.mvn.variance
        ).squeeze(-1)
        posterior_statistics = {"initial_entropy": initial_entropy}
        # Compute the posterior entropy term.
        conditional_posterior_with_noise = self.conditional_model.posterior(
            X.unsqueeze(-2).unsqueeze(-3), observation_noise=True
        )

        # `batch_shape x num_pareto_samples x q x 1 x M`
        post_mean = conditional_posterior_with_noise.mean.swapaxes(-4, -3)
        post_var_with_noise = conditional_posterior_with_noise.variance.clamp_min(
            CLAMP_LB
        ).swapaxes(-4, -3)
        # TODO: This computes the observation noise via a second evaluation of the
        #   posterior. This step could be done better.
        conditional_posterior = self.conditional_model.posterior(
            X.unsqueeze(-2).unsqueeze(-3), observation_noise=False
        )

        # `batch_shape x num_pareto_samples x q x num_samples_per_GP x M`
        post_var = conditional_posterior.variance.clamp_min(CLAMP_LB).swapaxes(-4, -3)
        obs_noise = (post_var_with_noise - post_var).clamp_min(CLAMP_LB)

        posterior_statistics["posterior_mean"] = post_mean.transpose(-1, -2)
        posterior_statistics["posterior_variance"] = post_var.transpose(-1, -2)
        posterior_statistics["observation_noise"] = obs_noise.transpose(-1, -2)
        posterior_statistics["posterior_with_noise"] = conditional_posterior_with_noise
        return posterior_statistics

    def _compute_lower_bound_information_gain(self, X: Tensor, return_parts: bool = False) -> Tensor:
        r"""Evaluates the lower bound information gain at the design points `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of `batch_shape` t-batches with `q`
            `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of acquisition values at the given design
            points `X`.
        """
        
        posterior_statistics = self._compute_posterior_statistics(X)
        initial_entropy = posterior_statistics["initial_entropy"]
        post_mean = posterior_statistics["posterior_mean"]
        post_var = posterior_statistics["posterior_variance"]
        obs_noise = posterior_statistics["observation_noise"]

        if self.estimation_type == "LB":
            conditional_entropy = _compute_entropy_upper_bound(
                optimal_values=self.optimal_outputs,
                mean=post_mean,
                variance=post_var,
                observation_noise=obs_noise,
                return_parts=True,
                only_diagonal=False,
            )

        elif self.estimation_type == "LB2":
            conditional_entropy = _compute_entropy_upper_bound(
                optimal_values=self.optimal_outputs,
                mean=post_mean,
                variance=post_var,
                observation_noise=obs_noise,
                only_diagonal=True,
                return_parts=True,
            )

        initial_entropy = initial_entropy.unsqueeze(-1).repeat(1,
                                1, self.num_samples)
        entropy_reduction = initial_entropy - conditional_entropy
        entropy_reduction = entropy_reduction.mean(-1).T.squeeze(-1)
        # This is a hack for PiES - needs to re-weight the conditional entropy terms

        if return_parts:
             return (initial_entropy, conditional_entropy)
        
        return entropy_reduction

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluates qLowerBoundMultiObjectiveJointEntropySearch at the design
        points `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of `batch_shape` t-batches with `q`
            `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of acquisition values at the given design
            points `X`.
        """
        return self._compute_lower_bound_information_gain(X)


def _compute_entropy_upper_bound(
    optimal_values: Tensor,
    mean: Tensor,
    variance: Tensor,
    observation_noise: Tensor,
    only_diagonal: bool = True,
    return_parts: bool = False,  # This is only for debugging
) -> Tensor:
    r"""Computes the entropy upper bound at the design points `X`. This is used for
    the JES-LB and MES-LB estimate. If `only_diagonal` is True, then this computes
    the entropy estimate for the JES-LB2 and MES-LB2.

    Args:
        hypercell_bounds: A `num_pareto_samples x 2 x J x M` -dim Tensor containing
            the box decomposition bounds, where `J` = max(num_boxes).
        mean: A `batch_shape x num_pareto_samples x q x 1 x M`-dim Tensor containing
            the posterior mean at X.
        variance: A `batch_shape x num_pareto_samples x q x 1 x M`-dim Tensor
            containing the posterior variance at X excluding observation noise.
        observation_noise: A `batch_shape x num_pareto_samples x q x 1 x M`-dim
            Tensor containing the observation noise at X.
        only_diagonal: If true, we only compute the diagonal elements of the variance.

    Returns:
        A `batch_shape x q`-dim Tensor of entropy estimate at the given design points
        `X`.
    """
    tkwargs = {"dtype": optimal_values.dtype, "device": optimal_values.device}
    CLAMP_LB = torch.finfo(tkwargs["dtype"]).eps

    mean = mean.unsqueeze(-2)
    variance = variance.unsqueeze(-2)
    observation_noise = observation_noise.unsqueeze(-2)
    variance_plus_noise = variance + observation_noise

    # Standardize the box decomposition bounds and compute normal quantities.
    # `batch_shape x num_pareto_samples x q x 2 x J x M`
    optimal_values = optimal_values.unsqueeze(
        1).repeat(1, mean.shape[1], 1, 1).unsqueeze(-2)
    g = ((optimal_values - mean) / torch.sqrt(variance))

    normal = Normal(torch.zeros_like(g), torch.ones_like(g))
    gcdf = normal.cdf(g)
    gpdf = torch.exp(normal.log_prob(g))
    g_times_gpdf = g * gpdf

    # Compute the differences between the upper and lower terms.
    # (NO NEED SINGLE-OBJECTIVE)
    Wjm = gcdf
    Vjm = g_times_gpdf
    Gjm = gpdf

    # Compute W.
    Wj = torch.exp(torch.sum(torch.log(Wjm), dim=-1, keepdims=True))
    W = torch.sum(Wj, dim=-2, keepdims=True).clamp_max(1.0)

    Cjm = Gjm / Wjm

    # First moment:
    Rjm = Cjm * Wj / W
    # `batch_shape x num_pareto_samples x q x 1 x M
    mom1 = mean - torch.sqrt(variance) * Rjm.sum(-2, keepdims=True)
    # diagonal weighted sum
    # `batch_shape x num_pareto_samples x q x 1 x M
    diag_weighted_sum = (Wj * variance * Vjm / Wjm / W).sum(-2, keepdims=True)

    if only_diagonal:
        # `batch_shape x num_pareto_samples x q x 1 x M`
        mean_squared = mean.pow(2)
        cross_sum = -2 * (mean * torch.sqrt(variance) * Rjm).sum(-2, keepdims=True)
        # `batch_shape x num_pareto_samples x q x 1 x M`
        mom2 = variance_plus_noise - diag_weighted_sum + cross_sum + mean_squared
        var = (mom2 - mom1.pow(2)).clamp_min(CLAMP_LB)

        # `batch_shape x num_pareto_samples x q
        log_det_term = 0.5 * torch.log(var).sum(dim=-1).squeeze(-1)
    else:
        # First moment x First moment
        # `batch_shape x num_pareto_samples x q x 1 x M x M
        cross_mom1 = torch.einsum("...i,...j->...ij", mom1, mom1)

        # Second moment:
        # `batch_shape x num_pareto_samples x q x 1 x M x M
        # firstly compute the general terms
        mom2_cross1 = -torch.einsum(
            "...i,...j->...ij", mean, torch.sqrt(variance) * Cjm
        )
        mom2_cross2 = -torch.einsum(
            "...i,...j->...ji", mean, torch.sqrt(variance) * Cjm
        )
        mom2_mean_squared = torch.einsum("...i,...j->...ij", mean, mean)

        mom2_weighted_sum = (
            (mom2_cross1 + mom2_cross2) * Wj.unsqueeze(-1) / W.unsqueeze(-1)
        ).sum(-3, keepdims=True)
        mom2_weighted_sum = mom2_weighted_sum + mom2_mean_squared

        # Compute the additional off-diagonal terms.
        mom2_off_diag = torch.einsum(
            "...i,...j->...ij", torch.sqrt(variance) * Cjm, torch.sqrt(variance) * Cjm
        )
        mom2_off_diag_sum = (mom2_off_diag * Wj.unsqueeze(-1) / W.unsqueeze(-1)).sum(
            -3, keepdims=True
        )

        # Compute the diagonal terms and subtract the diagonal computed before.
        init_diag = torch.diagonal(mom2_off_diag_sum, dim1=-2, dim2=-1)
        diag_weighted_sum = torch.diag_embed(
            variance_plus_noise - diag_weighted_sum - init_diag
        )
        mom2 = mom2_weighted_sum + mom2_off_diag_sum + diag_weighted_sum
        # Compute the variance
        var = (mom2 - cross_mom1).clamp_min(CLAMP_LB).squeeze(-3)

        # Jitter the diagonal.
        # The jitter is probably not needed here at all.
        jitter_diag = 1e-6 * torch.diag_embed(torch.ones(var.shape[:-1], **tkwargs))
        log_det_term = 0.5 * torch.logdet(var + jitter_diag)

    # Additional terms.
    M_plus_K = mean.shape[-1]
    add_term = 0.5 * M_plus_K * (1 + torch.log(torch.ones(1, **tkwargs) * 2 * pi))

    # `batch_shape x num_pareto_samples x q
    entropy = add_term + log_det_term
    add_nan = torch.sum(torch.isnan(add_term))
    log_det_nan = torch.sum(torch.isnan(log_det_term))

    if return_parts:
        return entropy
    
    return entropy.mean(-2)


class qExploitLowerBoundJointEntropySearch(qLowerBoundJointEntropySearch):
    r"""The acquisition function for the Joint Entropy Search, where the batches
    `q > 1` are supported through the lower bound formulation.

    This acquisition function computes the mutual information between the observation
    at a candidate point X and the Pareto optimal input-output pairs.
    """

    def __init__(
        self,
        model: Model,
        optimal_inputs: Tensor,
        optimal_outputs: Tensor,
        condition_noiseless: bool = True,
        maximize: bool = True,
        X_pending: Optional[Tensor] = None,
        exploit_fraction: float = 0.1,
        estimation_type: str = "LB2",
        num_samples: int = 64,
        **kwargs: Any,
    ) -> None:
        r"""gamma-exploit Joint entropy search acquisition function.

        Args:
            model: A fitted batch model with 'M + K' number of outputs. The
                first `M` corresponds to the objectives and the rest corresponds
                to the constraints. An input is feasible if f_k(x) <= 0 for the
                constraint functions f_k. The number `K` is specified the variable
                `num_constraints`.
            pareto_sets: A `num_pareto_samples x num_pareto_points x d`-dim Tensor.
            pareto_fronts: A `num_pareto_samples x num_pareto_points x M`-dim Tensor.
            hypercell_bounds:  A `num_pareto_samples x 2 x J x (M + K)`-dim
                Tensor containing the hyper-rectangle bounds for integration. In the
                unconstrained case, this gives the partition of the dominated space.
                In the constrained case, this gives the partition of the feasible
                dominated space union the infeasible space.
            condition_noiseless: whether or not to condition on noiseless or noisy optimal
            observations. This is the main differentiator between 
                "Joint Entropy Search for Multi-Objective Bayesian Optimization" (False) 
                "Joint Entropy Search for Maximally-Informed Bayesian Optimization"(True)
            maximize: If True, consider the problem a maximization problem.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
            estimation_type: A string to determine which entropy estimate is
                computed: "Noiseless", "Lower bound" or "Monte Carlo".
            only_diagonal: If true we only compute the diagonal elements of the
                variance for the `Lower bound` estimation strategy i.e. the LB2
                strategy.
            sampler: The sampler used if Monte Carlo is used to estimate the entropy.
                Defaults to 'SobolQMCNormalSampler(num_samples=64,
                collapse_batch_dims=True)'.
            num_samples: The number of Monte Carlo samples if using the default Monte
                Carlo sampler.
            num_constraints: The number of constraints.
        """
        super().__init__(
            model=model,
            optimal_inputs=optimal_inputs,
            optimal_outputs=optimal_outputs,
            condition_noiseless=condition_noiseless,
            maximize=maximize,
            X_pending=X_pending,
            exploit_fraction=exploit_fraction,
            estimation_type=estimation_type,
            num_samples=num_samples,
            **kwargs
        )
        self.exploit = torch.rand([1]) < exploit_fraction

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Compute joint entropy search at the design points `X`, iterating between
        greedy and non-greedy depending on the iteration.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of `batch_shape` t-batches with `1`
                `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of JES values at the given design points `X`.
        """

        if self.exploit:
            if is_fully_bayesian(self.model):
                X = X.squeeze(1)
                res = self.initial_model.posterior(X).mean.squeeze(-1).squeeze(-1).T
            else:
                res = self.initial_model.posterior(X).mean.squeeze(-1).squeeze(-1)
        else:
            res = self._compute_lower_bound_information_gain(X)

        return res
