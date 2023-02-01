# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


r"""Gaussian Process Regression models with fully Bayesian inference.

Fully Bayesian models use Bayesian inference over model hyperparameters, such
as lengthscales and noise variance, learning a posterior distribution for the
hyperparameters using the No-U-Turn-Sampler (NUTS). This is followed by
sampling a small set of hyperparameters (often ~16) from the posterior
that we will use for model predictions and for computing acquisition function
values. By contrast, our “standard” models (e.g.
`SingleTaskGP`) learn only a single best value for each hyperparameter using
MAP. The fully Bayesian method generally results in a better and more
well-calibrated model, but is more computationally intensive. For a full
description, see [Eriksson2021saasbo].

We use a lightweight PyTorch implementation of a Matern-5/2 kernel as there are
some performance issues with running NUTS on top of standard GPyTorch models.
The resulting hyperparameter samples are loaded into a batched GPyTorch model
after fitting.

References:

.. [Eriksson2021saasbo]
    D. Eriksson, M. Jankowiak. High-Dimensional Bayesian Optimization
    with Sparse Axis-Aligned Subspaces. Proceedings of the Thirty-
    Seventh Conference on Uncertainty in Artificial Intelligence, 2021.
"""


import math
import warnings
from abc import abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pyro
import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils import validate_input_scaling
from botorch.posteriors.fully_bayesian import FullyBayesianPosterior, MCMC_DIM
from gpytorch.constraints import GreaterThan
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.kernels.kernel import Distance, Kernel
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
import gpytorch
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.means.mean import Mean
from gpytorch.means import ZeroMean
from gpytorch.models.exact_gp import ExactGP
from linear_operator import settings
from torch import Tensor
from torch.linalg import inv, solve_triangular, cholesky
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.utils.errors import NotPSDError
from botorch.models.gp_regression import SingleTaskGP


MIN_INFERRED_NOISE_LEVEL = 1e-6


# TODO exponential kernel
def sqexp_kernel(X: Tensor, lengthscale: Tensor) -> Tensor:
    """Squared exponential kernel."""

    dist = compute_dists(X=X, lengthscale=lengthscale)
    exp_component = torch.exp(-torch.pow(dist, 2) / 2)
    return exp_component


def matern52_kernel(X: Tensor, lengthscale: Tensor) -> Tensor:
    """Matern-5/2 kernel."""
    nu = 5 / 2
    dist = compute_dists(X=X, lengthscale=lengthscale)
    exp_component = torch.exp(-math.sqrt(nu * 2) * dist)
    constant_component = (math.sqrt(5) * dist).add(1).add(5.0 / 3.0 * (dist**2))
    return constant_component * exp_component


def compute_dists(X: Tensor, lengthscale: Tensor) -> Tensor:
    """Compute kernel distances."""
    return Distance()._dist(
        X / lengthscale, X / lengthscale, postprocess=False, x1_eq_x2=True
    )


def reshape_and_detach(target: Tensor, new_value: Tensor) -> None:
    """Detach and reshape `new_value` to match `target`."""
    return new_value.detach().clone().view(target.shape).to(target)


def _psd_safe_pyro_mvn_sample(
    name: str, loc: Tensor, covariance_matrix: Tensor, obs: Tensor
) -> None:
    r"""Wraps the `pyro.sample` call in a loop to add an increasing series of jitter
    to the covariance matrix each time we get a LinAlgError.

    This is modelled after linear_operator's `psd_safe_cholesky`.
    """
    jitter = settings.cholesky_jitter.value(loc.dtype)
    max_tries = settings.cholesky_max_tries.value()
    for i in range(max_tries + 1):
        jitter_matrix = (
            torch.eye(
                covariance_matrix.shape[-1],
                device=covariance_matrix.device,
                dtype=covariance_matrix.dtype,
            )
            * jitter
        )
        jittered_covar = (
            covariance_matrix if i == 0 else covariance_matrix + jitter_matrix
        )
        try:
            pyro.sample(
                name,
                pyro.distributions.MultivariateNormal(
                    loc=loc,
                    covariance_matrix=jittered_covar,
                ),
                obs=obs,
            )
            return
        except (torch.linalg.LinAlgError, ValueError) as e:
            if isinstance(e, ValueError) and "satisfy the constraint" not in str(e):
                # Not-PSD can be also caught in Distribution.__init__ during parameter
                # validation, which raises a ValueError. Only catch those errors.
                raise e
            jitter = jitter * (10**i)
            warnings.warn(
                "Received a linear algebra error while sampling with Pyro. Adding a "
                f"jitter of {jitter} to the covariance matrix and retrying.",
                RuntimeWarning,
            )


def normal_tensor_likelihood(hp_tensor, train_X, train_Y, gp_kernel='matern'):
    outputscale = hp_tensor[0]
    noise = hp_tensor[1]
    lengthscales = hp_tensor[2:]
    return normal_log_likelihood(train_X, train_Y, outputscale, noise, lengthscales, gp_kernel=gp_kernel)


def normal_tensor_likelihood_noscale(hp_tensor, train_X, train_Y, gp_kernel='matern'):
    outputscale = torch.ones_like(hp_tensor[0])
    noise = hp_tensor[0]
    lengthscales = hp_tensor[1:]
    return normal_log_likelihood(train_X, train_Y, outputscale, noise, lengthscales, gp_kernel=gp_kernel)


def normal_log_likelihood(train_X, train_Y, outputscale, noise, lengthscales, gp_kernel='matern'):
    if gp_kernel == "matern":
        k_noiseless = matern52_kernel(X=train_X, lengthscale=lengthscales)
    elif gp_kernel == "rbf":
        k_noiseless = sqexp_kernel(X=train_X, lengthscale=lengthscales)
    else:
        raise ValueError('Not a valid kernel, choose matern or rbf.')
    k = outputscale * k_noiseless + noise * \
        torch.eye(train_X.shape[0], dtype=train_X.dtype, device=train_X.device)

    L_cholesky = cholesky(k)
    Lf_y = solve_triangular(L_cholesky, train_Y, upper=False)
    Lf_yy = solve_triangular(L_cholesky.T, Lf_y, upper=True)

    const = 0.5 * len(train_Y) * torch.log(Tensor([2 * torch.pi]))
    logdet = torch.sum(torch.log(torch.diag(L_cholesky)))
    # ensure this actually returns a scalar
    logprobs = 0.5 * torch.matmul(train_Y.T, Lf_yy)
    return -(const + logdet + logprobs)


class Sampler:
    pass


class EllipticalSliceSampler(Sampler):

    def __init__(self, prior, lnpdf, num_samples, pdf_params=(), warmup=0, thinning=1):
        r"""
        RETRIEVED FROM Wesley Maddox's github (and adapted ever so slightly): 
        https://github.com/wjmaddox/pytorch_ess/blob/main/pytorch_ess/elliptical_slice.py
        Implementation of elliptical slice sampling (Murray, Adams, & Mckay, 2010).
        The current implementation assumes that every parameter is log normally distributed.
        TODO: allow for normal (i.e. non-lognormal) priors.
        """

        # TODO allow for labeling of each dimension so that things don't have to come in the
        # order outputscale, noise, lengthscale
        self.n = prior.mean.nelement()
        self.prior = prior
        self.lnpdf = lnpdf
        self.num_samples = num_samples + warmup
        self.pdf_params = pdf_params
        self.warmup = warmup
        self.thinning = thinning
        self.f_priors = (prior.rsample(torch.Size(
            [self.num_samples])) - prior.mean.unsqueeze(0))
        self.num_mcmc_samples = int(num_samples / thinning)

    def untransform_sample(self, sample):
        return torch.exp(sample + self.prior.mean)

    def get_samples(self):
        raw_samples, likelihood = self.run()
        raw_samples = raw_samples
        thinned_samples = raw_samples[self.warmup::self.thinning, :]
        return self.untransform_sample(thinned_samples)

    def run(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.f_sampled = torch.zeros(
            self.num_samples, self.n, device=self.f_priors.device, dtype=self.f_priors.dtype)
        self.ell = torch.zeros(
            self.num_samples, 1, device=self.f_priors.device, dtype=self.f_priors.dtype)

        f_cur = torch.zeros_like(self.prior.mean)
        for ii in range(self.num_samples):
            if ii == 0:
                ell_cur = self.lnpdf(self.untransform_sample(f_cur), *self.pdf_params)
            else:
                # Retrieves the previous best - it is a markov chain =)
                # could probably do with some warmup and thinning for diversity then as well
                f_cur = self.f_sampled[ii - 1, :]
                ell_cur = self.ell[ii - 1, 0]

            next_f_prior = self.f_priors[ii, :]

            self.f_sampled[ii, :], self.ell[ii] = self.elliptical_slice(f_cur, next_f_prior,
                                                                        cur_lnpdf=ell_cur, pdf_params=self.pdf_params)

        return self.f_sampled, self.ell

    def elliptical_slice(
        self,
        initial_theta: torch.Tensor,
        prior: torch.Tensor,
        pdf_params: Tuple = (),
        cur_lnpdf: torch.Tensor = None,
        angle_range: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        D = len(initial_theta)
        if cur_lnpdf is None:
            cur_lnpdf = self.lnpdf(self.untransform_sample(initial_theta), *pdf_params)

        ## FORCING THE RIGHT PRIOR TO BE GIVEN ##
        # Set up the ellipse and the slice threshold
        if len(prior.shape) == 1:  # prior = prior sample
            nu = prior
        else:  # prior = cholesky decomp
            if not prior.shape[0] == D or not prior.shape[1] == D:
                raise IOError(
                    "Prior must be given by a D-element sample or DxD chol(Sigma)")
            nu = prior.transpose(-1, -2).matmul(torch.randn(
                prior.shape[:-2], 1, device=prior.device, dtype=prior.dtype))

        hh = torch.rand(1).log() + cur_lnpdf

        # Set up a bracket of angles and pick a first proposal.
        # "phi = (theta'-theta)" is a change in angle.
        if angle_range is None or angle_range == 0.:
            # Bracket whole ellipse with both edges at first proposed point
            phi = torch.rand(1) * 2. * math.pi
            phi_min = phi - 2. * math.pi
            phi_max = phi
        else:
            # Randomly center bracket on current point
            phi_min = -angle_range * torch.rand(1)
            phi_max = phi_min + angle_range
            phi = torch.rand(1) * (phi_max - phi_min) + phi_min

        # Slice sampling loop
        while True:
            # Compute xx for proposed angle difference and check if it's on the slice
            xx_prop = initial_theta * math.cos(phi) + nu * math.sin(phi)

            cur_lnpdf = self.lnpdf(self.untransform_sample(xx_prop), *pdf_params)
            if cur_lnpdf > hh:
                # New point is on slice, ** EXIT LOOP **
                break
            # Shrink slice to rejected point
            if phi > 0:
                phi_max = phi
            elif phi < 0:
                phi_min = phi
            else:
                raise RuntimeError(
                    'BUG DETECTED: Shrunk to current position and still not acceptable.')
            # Propose new angle difference
            phi = torch.rand(1) * (phi_max - phi_min) + phi_min

        return (xx_prop, cur_lnpdf)

    def load_mcmc_samples(self):
        samples = self.get_samples()
        samples_dict = {}
        samples_dict['outputscale'] = samples[: , 0]
        samples_dict['noise'] = samples[: , 1]
        samples_dict['lengthscale'] = samples[: , 2:]
        return samples_dict


class PyroModel:
    r"""
    Base class for a Pyro model; used to assist in learning hyperparameters.

    This class and its subclasses are not a standard BoTorch models; instead
    the subclasses are used as inputs to a `SaasFullyBayesianSingleTaskGP`,
    which should then have its hyperparameters fit with
    `fit_fully_bayesian_model_nuts`. (By default, its subclass `SaasPyroModel`
    is used).  A `PyroModel`’s `sample` method should specify lightweight
    PyTorch functionality, which will be used for fast model fitting with NUTS.
    The utility of `PyroModel` is in enabling fast fitting with NUTS, since we
    would otherwise need to use GPyTorch, which is computationally infeasible
    in combination with Pyro.

    :meta private:
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        """Set the training data.

        Args:
            train_X: Training inputs (n x d)
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1). Inferred if None.
        """
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar

    @abstractmethod
    def sample(self) -> None:
        r"""Sample from the model."""
        pass  # pragma: no cover

    @abstractmethod
    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor], **kwargs: Any
    ) -> Dict[str, Tensor]:
        """Post-process the final MCMC samples."""
        pass  # pragma: no cover

    @abstractmethod
    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        pass  # pragma: no cover


class SaasPyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(concentration=2.0, rate=0.15, **tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        k = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        k = outputscale * k + noise * torch.eye(self.train_X.shape[0], **tkwargs)
        _psd_safe_pyro_mvn_sample(
            name="Y",
            loc=mean.view(-1).expand(self.train_X.shape[0]),
            covariance_matrix=k,
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(
        self, concentration: float = 2.0, rate: float = 0.15, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the outputscale."""
        return pyro.sample(
            "outputscale",
            pyro.distributions.Gamma(
                torch.tensor(concentration, **tkwargs),
                torch.tensor(rate, **tkwargs),
            ),
        )

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ),
        )

    def sample_noise(self, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            return pyro.sample(
                "noise",
                pyro.distributions.Gamma(
                    torch.tensor(0.9, **tkwargs),
                    torch.tensor(10.0, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_lengthscale(
        self, dim: int, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        tausq = pyro.sample(
            "kernel_tausq",
            pyro.distributions.HalfCauchy(torch.tensor(alpha, **tkwargs)),
        )
        inv_length_sq = pyro.sample(
            "_kernel_inv_length_sq",
            pyro.distributions.HalfCauchy(torch.ones(dim, **tkwargs)),
        )
        inv_length_sq = pyro.deterministic(
            "kernel_inv_length_sq", tausq * inv_length_sq
        )
        lengthscale = pyro.deterministic(
            "lengthscale",
            (1.0 / inv_length_sq).sqrt(),
        )
        return lengthscale

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        inv_length_sq = (
            mcmc_samples["kernel_tausq"].unsqueeze(-1)
            * mcmc_samples["_kernel_inv_length_sq"]
        )
        mcmc_samples["lengthscale"] = (1.0 / inv_length_sq).sqrt()
        # Delete `kernel_tausq` and `_kernel_inv_length_sq` since they aren't loaded
        # into the final model.
        del mcmc_samples["kernel_tausq"], mcmc_samples["_kernel_inv_length_sq"]
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["mean"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=mcmc_samples["outputscale"],
        )
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )
        return mean_module, covar_module, likelihood


class SaasFullyBayesianSingleTaskGP(ExactGP, BatchedMultiOutputGPyTorchModel):
    r"""A fully Bayesian single-task GP model with the SAAS prior.

    This model assumes that the inputs have been normalized to [0, 1]^d and thatcdcd
    the output has been standardized to have zero mean and unit variance. You can
    either normalize and standardize the data before constructing the model or use
    an `input_transform` and `outcome_transform`. The SAAS model [Eriksson2021saasbo]_
    with a Matern-5/2 kernel is used by default.

    You are expected to use `fit_fully_bayesian_model_nuts` to fit this model as it
    isn't compatible with `fit_gpytorch_model`.

    Example:
        >>> saas_gp = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(saas_gp)
        >>> posterior = saas_gp.posterior(test_X)
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        pyro_model: Optional[PyroModel] = None,
    ) -> None:
        r"""Initialize the fully Bayesian single-task GP model.

        Args:
            train_X: Training inputs (n x d)
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1). Inferred if None.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                forward pass.
            pyro_model: Optional `PyroModel`, defaults to `SaasPyroModel`.
        """
        if not (
            train_X.ndim == train_Y.ndim == 2
            and len(train_X) == len(train_Y)
            and train_Y.shape[-1] == 1
        ):
            raise ValueError(
                "Expected train_X to have shape n x d and train_Y to have shape n x 1"
            )
        if train_Yvar is not None:
            if train_Y.shape != train_Yvar.shape:
                raise ValueError(
                    "Expected train_Yvar to be None or have the same shape as train_Y"
                )
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        validate_input_scaling(
            train_X=transformed_X, train_Y=train_Y, train_Yvar=train_Yvar
        )
        self._num_outputs = train_Y.shape[-1]
        self._input_batch_shape = train_X.shape[:-2]
        if train_Yvar is not None:  # Clamp after transforming
            train_Yvar = train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL)

        X_tf, Y_tf, _ = self._transform_tensor_args(X=train_X, Y=train_Y)
        super().__init__(
            train_inputs=X_tf, train_targets=Y_tf, likelihood=GaussianLikelihood()
        )
        self.mean_module = None
        self.covar_module = None
        self.likelihood = None
        if pyro_model is None:
            pyro_model = SaasPyroModel()
        pyro_model.set_inputs(
            train_X=transformed_X, train_Y=train_Y, train_Yvar=train_Yvar
        )
        self.pyro_model = pyro_model
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform

    def _check_if_fitted(self):
        r"""Raise an exception if the model hasn't been fitted."""
        if self.covar_module is None:
            raise RuntimeError(
                "Model has not been fitted. You need to call "
                "`fit_fully_bayesian_model_nuts` to fit the model."
            )

    @property
    def median_lengthscale(self) -> Tensor:
        r"""Median lengthscales across the MCMC samples."""
        self._check_if_fitted()
        lengthscale = self.covar_module.base_kernel.lengthscale.clone()
        return lengthscale.median(0).values.squeeze(0)

    @property
    def num_mcmc_samples(self) -> int:
        r"""Number of MCMC samples in the model."""
        self._check_if_fitted()
        return len(self.covar_module.outputscale)

    @property
    def batch_shape(self) -> torch.Size:
        r"""Batch shape of the model, equal to the number of MCMC samples.
        Note that `SaasFullyBayesianSingleTaskGP` does not support batching
        over input data at this point."""
        return torch.Size([self.num_mcmc_samples])

    @property
    def _aug_batch_shape(self) -> torch.Size:
        r"""The batch shape of the model, augmented to include the output dim."""
        aug_batch_shape = self.batch_shape
        if self.num_outputs > 1:
            aug_batch_shape += torch.Size([self.num_outputs])
        return aug_batch_shape

    def train(self, mode: bool = True) -> None:
        r"""Puts the model in `train` mode."""
        super().train(mode=mode)
        if mode:
            self.mean_module = None
            self.covar_module = None
            self.likelihood = None

    def load_mcmc_samples(self, mcmc_samples: Dict[str, Tensor]) -> None:
        r"""Load the MCMC hyperparameter samples into the model.

        This method will be called by `fit_fully_bayesian_model_nuts` when the model
        has been fitted in order to create a batched SingleTaskGP model.
        """
        (
            self.mean_module,
            self.covar_module,
            self.likelihood,
        ) = self.pyro_model.load_mcmc_samples(mcmc_samples=mcmc_samples)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        r"""Custom logic for loading the state dict.

        The standard approach of calling `load_state_dict` currently doesn't play well
        with the `SaasFullyBayesianSingleTaskGP` since the `mean_module`, `covar_module`
        and `likelihood` aren't initialized until the model has been fitted. The reason
        for this is that we don't know the number of MCMC samples until NUTS is called.
        Given the state dict, we can initialize a new model with some dummy samples and
        then load the state dict into this model. This currently only works for a
        `SaasPyroModel` and supporting more Pyro models likely requires moving the model
        construction logic into the Pyro model itself.
        """

        if not isinstance(self.pyro_model, SaasPyroModel):
            raise NotImplementedError("load_state_dict only works for SaasPyroModel")
        raw_mean = state_dict["mean_module.raw_constant"]
        num_mcmc_samples = len(raw_mean)
        dim = self.pyro_model.train_X.shape[-1]
        tkwargs = {"device": raw_mean.device, "dtype": raw_mean.dtype}
        # Load some dummy samples
        mcmc_samples = {
            "mean": torch.ones(num_mcmc_samples, **tkwargs),
            "lengthscale": torch.ones(num_mcmc_samples, dim, **tkwargs),
            "outputscale": torch.ones(num_mcmc_samples, **tkwargs),
        }
        if self.pyro_model.train_Yvar is None:
            mcmc_samples["noise"] = torch.ones(num_mcmc_samples, **tkwargs)
        (
            self.mean_module,
            self.covar_module,
            self.likelihood,
        ) = self.pyro_model.load_mcmc_samples(mcmc_samples=mcmc_samples)
        # Load the actual samples from the state dict
        super().load_state_dict(state_dict=state_dict, strict=strict)

    def forward(self, X: Tensor) -> MultivariateNormal:
        """
        Unlike in other classes' `forward` methods, there is no `if self.training`
        block, because it ought to be unreachable: If `self.train()` has been called,
        then `self.covar_module` will be None, `check_if_fitted()` will fail, and the
        rest of this method will not run.
        """
        self._check_if_fitted()
        x = X.unsqueeze(MCMC_DIM)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    # pyre-ignore[14]: Inconsistent override
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> FullyBayesianPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q x m`).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `FullyBayesianPosterior` object. Includes observation noise if specified.
        """
        self._check_if_fitted()
        posterior = super().posterior(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )
        posterior = FullyBayesianPosterior(distribution=posterior.distribution)
        return posterior


class SliceFullyBayesianSingleTaskGP(SingleTaskGP):
    r"""A fully Bayesian single-task GP model where the sampler is passed in.

    This model assumes that the inputs have been normalized to [0, 1]^d and that
    the output has been standardized to have zero mean and unit variance. You can
    either normalize and standardize the data before constructing the model or use
    an `input_transform` and `outcome_transform`.


    Example:
        >>> saas_gp = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(saas_gp)
        >>> posterior = saas_gp.posterior(test_X)
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        prior: MultivariateNormal,
        train_Yvar: Optional[Tensor] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        num_samples: int = 256,
        warmup: int = 128,
        thinning: int = 16
    ):
        if train_Yvar:
            raise ValueError('SliceGP does not support fixed noise for now.')
        self.sampler = EllipticalSliceSampler(
            prior,
            lnpdf=normal_tensor_likelihood,
            pdf_params=(train_X, train_Y),
            num_samples=num_samples,
            warmup=warmup,
            thinning=thinning,
        )
        num_mcmc_samples = self.sampler.num_mcmc_samples
        train_X = train_X.unsqueeze(0).expand(num_mcmc_samples, train_X.shape[0], -1)
        train_Y = train_Y.unsqueeze(0).expand(num_mcmc_samples, train_Y.shape[0], -1)
        super().__init__(train_X, train_Y)
        self.covar_module = None

    def _check_if_fitted(self):
        r"""Raise an exception if the model hasn't been fitted."""
        if self.covar_module is None:
            raise RuntimeError(
                "Model has not been fitted. You need to call "
                "`fit_fully_bayesian_model_nuts` to fit the model."
            )

    def forward(self, X: Tensor) -> MultivariateNormal:
        """
        Unlike in other classes' `forward` methods, there is no `if self.training`
        block, because it ought to be unreachable: If `self.train()` has been called,
        then `self.covar_module` will be None, `check_if_fitted()` will fail, and the
        rest of this method will not run.
        """
        self._check_if_fitted()
        return super().forward(X)

    def fit(self) -> None:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        mcmc_samples = self.sampler.load_mcmc_samples()
        tkwargs = {"dtype": self.train_targets.dtype,
                   "device": self.train_targets.device}
        num_mcmc_samples, ard_num_dims = mcmc_samples["lengthscale"].shape
        batch_shape = torch.Size([num_mcmc_samples])
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)

        # The input data should come in with zero mean, so this will just be zero
        # Could probably make this into a ZeroMean, too

        mean_module = ConstantMean(batch_shape=batch_shape)
        likelihood = GaussianLikelihood(
            batch_shape=batch_shape,
            noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
        ).to(**tkwargs)
        likelihood.noise_covar.noise = reshape_and_detach(
            target=likelihood.noise_covar.noise,
            new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
        )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=mcmc_samples["outputscale"],
        )

        # if we use MCMC over the mean, retrieve it. Else, set GP to be zero-mean.
        if mcmc_samples.get("mean", False):
            mean_module.constant.data = reshape_and_detach(
                target=mean_module.constant.data,
                new_value=mcmc_samples["mean"],
            )
        else:
            mean_module.constant.data = reshape_and_detach(
                target=mean_module.constant.data,
                new_value=torch.zeros(num_mcmc_samples),
            )
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.likelihood = likelihood

    def train(self, mode: bool = True) -> None:
        r"""Puts the model in `train` mode."""
        super().train(mode=mode)
        if mode:
            self.mean_module = None
            self.covar_module = None
            self.likelihood = None

    # pyre-ignore[14]: Inconsistent override
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> FullyBayesianPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q x m`).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `FullyBayesianPosterior` object. Includes observation noise if specified.
        """
        self._check_if_fitted()
        posterior = super().posterior(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )
        posterior = FullyBayesianPosterior(distribution=posterior.distribution)
        return posterior


class ActiveLearningPyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(**tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        k = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        k = outputscale * k + noise * torch.eye(self.train_X.shape[0], **tkwargs)
        _psd_safe_pyro_mvn_sample(
            name="Y",
            loc=mean.view(-1).expand(self.train_X.shape[0]),
            covariance_matrix=k,
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(
        self, mean: float = 0.0, variance: float = 1e-4, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the outputscale."""
        #return pyro.sample(
        #    "outputscale",
        #    pyro.distributions.LogNormal(
        #        torch.tensor(mean, **tkwargs),
        #        torch.tensor(variance, **tkwargs),
        #    ),
        #)
        return torch.tensor([1.0], **tkwargs)

    def sample_noise(self, mean: float = -1.0, variance: float = 3, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            r"""Sample the outputscale."""
            return pyro.sample(
                "noise",
                pyro.distributions.LogNormal(
                    torch.tensor(mean, **tkwargs),
                    torch.tensor(variance ** 0.5, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        return torch.tensor([0.0], **tkwargs)

    def sample_lengthscale(
        self, dim: int, mean: float = 0.0, variance: float = 3, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.LogNormal(
                torch.tensor(torch.ones(dim) * mean, **tkwargs),
                torch.tensor(torch.ones(dim) * variance ** 0.5, **tkwargs),
            ),
        )
        return lengthscale


    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ZeroMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=torch.ones_like(mcmc_samples["noise"]),
        )

        return mean_module, covar_module, likelihood  


class ActiveLearningWithOutputscalePyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(**tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        k = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        k = outputscale * k + noise * torch.eye(self.train_X.shape[0], **tkwargs)
        _psd_safe_pyro_mvn_sample(
            name="Y",
            loc=mean.view(-1).expand(self.train_X.shape[0]),
            covariance_matrix=k,
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(self, mean: float = 0.0, variance: float = 3, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        r"""Sample the outputscale."""
        return pyro.sample(
            "outputscale",
            pyro.distributions.LogNormal(
                torch.tensor(mean, **tkwargs),
                torch.tensor(variance ** 0.5, **tkwargs),
            ),
        )

    def sample_noise(self, mean: float = 0.0, variance: float = 3, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            r"""Sample the outputscale."""
            return pyro.sample(
                "noise",
                pyro.distributions.LogNormal(
                    torch.tensor(mean, **tkwargs),
                    torch.tensor(variance ** 0.5, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ),
        )

    def sample_lengthscale(
        self, dim: int, mean: float = 0.0, variance: float = 3, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.LogNormal(
                torch.ones(dim).to(**tkwargs) * mean,
                torch.ones(dim).to(**tkwargs) * variance ** 0.5
            ),
        )
        return lengthscale


    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=mcmc_samples["outputscale"]
        )
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )

        return mean_module, covar_module, likelihood    


class WideActiveLearningWithOutputscalePyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(**tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        k = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        k = outputscale * k + noise * torch.eye(self.train_X.shape[0], **tkwargs)
        _psd_safe_pyro_mvn_sample(
            name="Y",
            loc=mean.view(-1).expand(self.train_X.shape[0]),
            covariance_matrix=k,
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(self, mean: float = 1.0, variance: float = 9, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        r"""Sample the outputscale."""
        return pyro.sample(
            "outputscale",
            pyro.distributions.LogNormal(
                torch.tensor(mean, **tkwargs),
                torch.tensor(variance ** 0.5, **tkwargs),
            ),
        )

    def sample_noise(self, mean: float = 1.0, variance: float = 9, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            r"""Sample the outputscale."""
            return pyro.sample(
                "noise",
                pyro.distributions.LogNormal(
                    torch.tensor(mean, **tkwargs),
                    torch.tensor(variance ** 0.5, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ),
        )

    def sample_lengthscale(
        self, dim: int, mean: float = 1.0, variance: float = 9, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.LogNormal(
                torch.ones(dim).to(**tkwargs) * mean,
                torch.ones(dim).to(**tkwargs) * variance ** 0.5
            ),
        )
        return lengthscale


    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=mcmc_samples["outputscale"]
        )
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )
        return mean_module, covar_module, likelihood    


class BadActiveLearningWithOutputscalePyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(**tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        k = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        k = outputscale * k + noise * torch.eye(self.train_X.shape[0], **tkwargs)
        _psd_safe_pyro_mvn_sample(
            name="Y",
            loc=mean.view(-1).expand(self.train_X.shape[0]),
            covariance_matrix=k,
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(self, mean: float = 4.0, variance: float = 4, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        r"""Sample the outputscale."""
        return pyro.sample(
            "outputscale",
            pyro.distributions.LogNormal(
                torch.tensor(mean, **tkwargs),
                torch.tensor(variance ** 0.5, **tkwargs),
            ),
        )

    def sample_noise(self, mean: float = 4.0, variance: float = 4, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            r"""Sample the outputscale."""
            return pyro.sample(
                "noise",
                pyro.distributions.LogNormal(
                    torch.tensor(mean, **tkwargs),
                    torch.tensor(variance ** 0.5, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ),
        )

    def sample_lengthscale(
        self, dim: int, mean: float = 4.0, variance: float = 4.0, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.LogNormal(
                torch.ones(dim).to(**tkwargs) * mean,
                torch.ones(dim).to(**tkwargs) * variance ** 0.5
            ),
        )
        return lengthscale


    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=mcmc_samples["outputscale"]
        )
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )
        return mean_module, covar_module, likelihood    


class SingleTaskPyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(concentration=2.0, rate=0.15, **tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        k = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        k = outputscale * k + noise * torch.eye(self.train_X.shape[0], **tkwargs)
        _psd_safe_pyro_mvn_sample(
            name="Y",
            loc=mean.view(-1).expand(self.train_X.shape[0]),
            covariance_matrix=k,
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(
        self, concentration: float = 2.0, rate: float = 0.15, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the outputscale."""
        return pyro.sample(
            "outputscale",
            pyro.distributions.Gamma(
                torch.tensor(concentration, **tkwargs),
                torch.tensor(rate, **tkwargs),
            ),
        )

    def sample_noise(self, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            return pyro.sample(
                "noise",
                pyro.distributions.Gamma(
                    torch.tensor(1.1, **tkwargs),
                    torch.tensor(0.05, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        return torch.tensor([0.0], **tkwargs)

    def sample_lengthscale(
        self, dim: int, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.Gamma(
                (torch.ones(dim) * 3.0).to(**tkwargs),
                (torch.ones(dim) * 6.0).to(**tkwargs),
            ),
        )
        return lengthscale


    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ZeroMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=mcmc_samples["outputscale"],
        )

        return mean_module, covar_module, likelihood


class SingleTaskMeanPyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(concentration=2.0, rate=0.15, **tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        k = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        k = outputscale * k + noise * torch.eye(self.train_X.shape[0], **tkwargs)
        _psd_safe_pyro_mvn_sample(
            name="Y",
            loc=mean.view(-1).expand(self.train_X.shape[0]),
            covariance_matrix=k,
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(
        self, concentration: float = 2.0, rate: float = 0.15, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the outputscale."""
        return pyro.sample(
            "outputscale",
            pyro.distributions.Gamma(
                torch.tensor(concentration, **tkwargs),
                torch.tensor(rate, **tkwargs),
            ),
        )

    def sample_noise(self, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            return pyro.sample(
                "noise",
                pyro.distributions.Gamma(
                    torch.tensor(1.1, **tkwargs),
                    torch.tensor(0.05, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ),
        )

    def sample_lengthscale(
        self, dim: int, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        lengthscale = pyro.sample(
            "lengthscale",
            pyro.distributions.Gamma(
                (torch.ones(dim) * 3.0).to(**tkwargs),
                (torch.ones(dim) * 6.0).to(**tkwargs),
            ),
        )
        return lengthscale


    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["lengthscale"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=mcmc_samples["outputscale"],
        )
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )
        return mean_module, covar_module, likelihood

class PyroFullyBayesianSingleTaskGP(SingleTaskGP):
    r"""A fully Bayesian single-task GP model where the sampler is passed in.

    This model assumes that the inputs have been normalized to [0, 1]^d and that
    the output has been standardized to have zero mean and unit variance. You can
    either normalize and standardize the data before constructing the model or use
    an `input_transform` and `outcome_transform`.

    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        pyro_model: PyroModel,
        train_Yvar: Optional[Tensor] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        num_samples: int = 256,
        warmup: int = 128,
        thinning: int = 16,

    ):
        self.pyro_model = pyro_model()
        self.pyro_model.set_inputs(train_X, train_Y)
        num_mcmc_samples = num_mcmc_samples = int(num_samples / thinning)
        train_X = train_X.unsqueeze(0).expand(num_mcmc_samples, train_X.shape[0], -1)
        train_Y = train_Y.unsqueeze(0).expand(num_mcmc_samples, train_Y.shape[0], -1)
        super().__init__(train_X, train_Y)
        self.covar_module = None 
        self.num_samples = num_samples
        self.warmup = warmup
        self.thinning = thinning  

    def _check_if_fitted(self):
        r"""Raise an exception if the model hasn't been fitted."""
        if self.covar_module is None:
            raise RuntimeError(
                "Model has not been fitted. You need to call "
                "`fit_fully_bayesian_model_nuts` to fit the model."
            )

    def forward(self, X: Tensor) -> MultivariateNormal:
        """
        Unlike in other classes' `forward` methods, there is no `if self.training`
        block, because it ought to be unreachable: If `self.train()` has been called,
        then `self.covar_module` will be None, `check_if_fitted()` will fail, and the
        rest of this method will not run.
        """
        self._check_if_fitted()
        return super().forward(X)

    def fit(self) -> None:
        r"""Load the MCMC hyperparameter samples into the model.

        This method will be called by `fit_fully_bayesian_model_nuts` when the model
        has been fitted in order to create a batched SingleTaskGP model.
        """
        from botorch.fit import fit_fully_bayesian_model_nuts
        fit_fully_bayesian_model_nuts(
            model=self,
            warmup_steps=self.warmup,
            num_samples=self.num_samples,
            thinning=self.thinning,
        )
    
    def load_mcmc_samples(self, mcmc_samples) -> None:
        (
            self.mean_module,
            self.covar_module,
            self.likelihood,
        ) = self.pyro_model.load_mcmc_samples(mcmc_samples=mcmc_samples)

        # The input data should come in with zero mean, so this will just be zero
        # Could probably make this into a ZeroMean, too

    def train(self, mode: bool = True) -> None:
        r"""Puts the model in `train` mode."""
        super().train(mode=mode)
        if mode:
            self.mean_module = None
            self.covar_module = None
            self.likelihood = None

    # pyre-ignore[14]: Inconsistent override
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> FullyBayesianPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q x m`).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `FullyBayesianPosterior` object. Includes observation noise if specified.
        """
        self._check_if_fitted()
        posterior = super().posterior(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )
        posterior = FullyBayesianPosterior(distribution=posterior.distribution)
        return posterior