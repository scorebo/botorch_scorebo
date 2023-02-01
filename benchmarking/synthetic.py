import math
import torch
import numpy as np
from torch import Tensor

from botorch.test_functions.synthetic import SyntheticTestFunction, Branin



class Gramacy1(SyntheticTestFunction):

    def __init__(
        self,
        noise_std: float = 0,
        negate: bool = True
    ) -> None:
        self.dim = 1
        self._bounds = [(0.5, 2.5)]
        super().__init__(noise_std=noise_std, negate=negate, bounds=self._bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return (torch.sin(10 * math.pi * X) / (2 * X) + torch.pow(X - 1, 4)).flatten()


class Gramacy2(SyntheticTestFunction):

    def __init__(
        self,
        noise_std: float = 0,
        negate: bool = True
    ) -> None:
        self.dim = 2
        self._bounds = [(-2.0, 6.0), (-2.0, 6.0)]
        super().__init__(noise_std=noise_std, negate=negate, bounds=self._bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return (X[:, 0] * torch.exp(-torch.pow(X[:, 0], 2) - torch.pow(X[:, 1], 2))).flatten()


class Higdon(SyntheticTestFunction):

    def __init__(
        self,
        noise_std: float = 0,
        negate: bool = True
    ) -> None:
        self.dim = 1
        self._bounds = [(0.0, 20.0)]
        super().__init__(noise_std=noise_std, negate=negate, bounds=self._bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        x = X.detach().numpy()
        output = np.piecewise(x,
                              [x < 10, x >= 10],
                              [lambda xx: np.sin(np.pi * xx / 5) + 0.2 * np.cos(4 * np.pi * xx / 5),
                                  lambda xx: xx / 10 - 1])
        return Tensor(output).to(X.device).flatten()


class Ishigami(SyntheticTestFunction):
    def __init__(
        self,
        noise_std: float = 0,
        negate: bool = True
    ) -> None:
        self.dim = 3
        self._bounds = [(-3.141527, 3.141527)] * self.dim
        super().__init__(noise_std=noise_std, negate=negate, bounds=self._bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        a = 7
        b = 0.1
        return (torch.sin(X[:, 0]) + a * torch.pow(torch.sin(X[:, 1]), 2)
             + b * torch.pow(X[:, 2], 4) * torch.sin(X[:, 0])).flatten()


class ModBranin(Branin):

    def __init__(
        self,
        noise_std: float = 0,
        negate: bool = True
    ) -> None:
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        branin_vals = super().evaluate_true(X)
        valley_value =  self.fourth_valley(X) 
        return branin_vals + valley_value

    def fourth_valley(self, X: Tensor) -> Tensor:
        valley_center = Tensor([5, 11]).unsqueeze(0)
        valley_value =  -100 * torch.exp(-0.20216 * torch.sum(torch.pow(X - valley_center, 2), dim=-1))
        return valley_value


class MichM2(SyntheticTestFunction):
    r"""Michalewicz synthetic test function.

    d-dim function (usually evaluated on hypercube [0, pi]^d):

        M(x) = sum_{i=1}^d sin(x_i) (sin(i x_i^2 / pi)^20)
    """

    def __init__(
        self,
        dim=2,
        noise_std: float = 0.0,
        negate: bool = False,
        bounds: Tensor = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = dim
        self._bounds = [(0.0, math.pi) for _ in range(self.dim)]
        optvals = {2: -1.80130341, 5: -4.687658, 10: -9.66015}
        optimizers = {2: [(2.20290552, 1.57079633)]}
        self._optimal_value = optvals.get(self.dim)
        self._optimizers = optimizers.get(self.dim)
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)
        self.register_buffer(
            "i", torch.tensor(tuple(range(1, self.dim + 1)), dtype=torch.float)
        )

    @property
    def optimizers(self) -> Tensor:
        if self.dim in (5, 10):
            raise NotImplementedError()
        return super().optimizers

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        m = 2
        return -(
            torch.sum(
                torch.sin(X) * torch.sin(self.i * X**2 / math.pi) ** (2 * m), dim=-1
            )
        )