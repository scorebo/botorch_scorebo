#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Core abstractions and generic optimizers."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import auto, Enum
from itertools import count
from sys import maxsize
from time import monotonic
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from botorch.optim.closures import NdarrayOptimizationClosure
from botorch.optim.utils import get_bounds_as_ndarray
from numpy import asarray, float64 as np_float64, ndarray
from scipy.optimize import minimize
from torch import Tensor
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:  # pragma: no cover
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # pragma: no cover


_LBFGSB_MAXITER_MAXFUN_REGEX = re.compile(  # regex for maxiter and maxfun messages
    "TOTAL NO. of (ITERATIONS REACHED LIMIT|f AND g EVALUATIONS EXCEEDS LIMIT)"
)


class OptimizationStatus(int, Enum):
    RUNNING = auto()  # incomplete
    SUCCESS = auto()  # optimizer converged
    FAILURE = auto()  # terminated abnormally
    STOPPED = auto()  # stopped due to user provided criterion


@dataclass
class OptimizationResult:
    step: int
    fval: Union[float, int]
    status: OptimizationStatus
    runtime: Optional[float] = None
    message: Optional[str] = None


def scipy_minimize(
    closure: Union[
        Callable[[], Tuple[Tensor, Sequence[Optional[Tensor]]]],
        NdarrayOptimizationClosure,
    ],
    parameters: Dict[str, Tensor],
    bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    callback: Optional[Callable[[Dict[str, Tensor], OptimizationResult], None]] = None,
    x0: Optional[ndarray] = None,
    method: str = "L-BFGS-B",
    options: Optional[Dict[str, Any]] = None,
) -> OptimizationResult:
    r"""Generic scipy.optimize.minimize-based optimization routine.

    Args:
        closure: Callable that returns a tensor and an iterable of gradient tensors or
            NdarrayOptimizationClosure instance.
        parameters: A dictionary of tensors to be optimized.
        bounds: A dictionary mapping parameter names to lower and upper bounds.
        callback: A callable taking `parameters` and an OptimizationResult as arguments.
        x0: An optional initialization vector passed to scipy.optimize.minimize.
        method: Solver type, passed along to scipy.minimize.
        options: Dictionary of solver options, passed along to scipy.minimize.

    Returns:
        An OptimizationResult summarizing the final state of the run.
    """
    start_time = monotonic()
    wrapped_closure = (
        closure
        if isinstance(closure, NdarrayOptimizationClosure)
        else NdarrayOptimizationClosure(closure, parameters)
    )
    if bounds is None:
        bounds_np = None
    else:
        bounds_np = get_bounds_as_ndarray(parameters, bounds)

    if callback is None:
        wrapped_callback = None
    else:
        call_counter = count(1)  # callbacks are typically made at the end of each iter

        def wrapped_callback(x: ndarray):
            result = OptimizationResult(
                step=next(call_counter),
                fval=float(wrapped_closure(x)[0]),
                status=OptimizationStatus.RUNNING,
                runtime=monotonic() - start_time,
            )
            return callback(parameters, result)  # pyre-ignore [29]

    raw = minimize(
        wrapped_closure,
        wrapped_closure.state if x0 is None else x0.astype(np_float64, copy=False),
        jac=True,
        bounds=bounds_np,
        method=method,
        options=options,
        callback=wrapped_callback,
    )

    # Post-processing and outcome handling
    wrapped_closure.state = asarray(raw.x)  # set parameter state to optimal values
    msg = raw.message if isinstance(raw.message, str) else raw.message.decode("ascii")
    if raw.success:
        status = OptimizationStatus.SUCCESS
    else:
        status = (  # Check whether we stopped due to reaching maxfun or maxiter
            OptimizationStatus.STOPPED
            if _LBFGSB_MAXITER_MAXFUN_REGEX.search(msg)
            else OptimizationStatus.FAILURE
        )

    return OptimizationResult(
        fval=raw.fun,
        step=raw.nit,
        status=status,
        message=msg,
        runtime=monotonic() - start_time,
    )


def torch_minimize(
    closure: Callable[[], Tuple[Tensor, Sequence[Optional[Tensor]]]],
    parameters: Dict[str, Tensor],
    bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    callback: Optional[Callable[[Dict[str, Tensor], OptimizationResult], None]] = None,
    optimizer: Union[Optimizer, Callable[[List[Tensor]], Optimizer]] = Adam,
    scheduler: Optional[Union[LRScheduler, Callable[[Optimizer], LRScheduler]]] = None,
    step_limit: Optional[int] = None,
    stopping_criterion: Optional[Callable[[Tensor], bool]] = None,
) -> OptimizationResult:
    r"""Generic torch.optim-based optimization routine.

    Args:
        closure: Callable that returns a tensor and an iterable of gradient tensors.
            Responsible for setting relevant parameters' `grad` attributes.
        parameters: A dictionary of tensors to be optimized.
        bounds: An optional dictionary of bounds for elements of `parameters`.
        callback: A callable taking `parameters` and an OptimizationResult as arguments.
        step_limit: Integer specifying a maximum number of optimization steps.
            One of `step_limit` or `stopping_criterion` must be passed.
        stopping_criterion: A StoppingCriterion for the optimization loop.
        optimizer: A `torch.optim.Optimizer` instance or a factory that takes
            a list of parameters and returns an `Optimizer` instance.
        scheduler: A `torch.optim.lr_scheduler._LRScheduler` instance or a factory
            that takes a `Optimizer` instance and returns a `_LRSchedule` instance.

    Returns:
        An OptimizationResult summarizing the final state of the run.
    """
    start_time = monotonic()
    if step_limit is None:
        if stopping_criterion is None:
            raise RuntimeError("No termination conditions were given.")
        step_limit = maxsize

    if not isinstance(optimizer, Optimizer):
        optimizer = optimizer(list(parameters.values()))

    if not (scheduler is None or isinstance(scheduler, LRScheduler)):
        scheduler = scheduler(optimizer)

    _bounds = (
        {}
        if bounds is None
        else {name: limits for name, limits in bounds.items() if name in parameters}
    )
    result: OptimizationResult
    for step in range(step_limit):
        fval, _ = closure()
        result = OptimizationResult(
            step=step,
            fval=fval.detach().cpu().item(),
            status=OptimizationStatus.RUNNING,
            runtime=monotonic() - start_time,
        )

        # TODO: Update stopping_criterion API to return a message.
        if stopping_criterion and stopping_criterion(fval):
            result.status = OptimizationStatus.STOPPED
            result.message = "`torch_minimize` stopped due to `stopping_criterion`."

        if callback:
            callback(parameters, result)

        if result.status != OptimizationStatus.RUNNING:
            break

        optimizer.step()
        for name, (lower, upper) in _bounds.items():
            parameters[name].data = parameters[name].clamp(min=lower, max=upper)

        if scheduler:
            scheduler.step()

    if result.status != OptimizationStatus.RUNNING:
        return replace(result, runtime=monotonic() - start_time)

    # Account for final parameter update when stopping due to step_limit
    return OptimizationResult(
        step=step + 1,
        fval=closure()[0].detach().cpu().item(),
        status=OptimizationStatus.STOPPED,
        runtime=monotonic() - start_time,
        message=f"`torch_minimize` stopped after reaching step_limit={step_limit}.",
    )
