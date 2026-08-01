import math
from typing import List, Optional, Tuple
import torch
from torch import Tensor

from botorch.test_functions.synthetic import SyntheticTestFunction


class Ishigami(SyntheticTestFunction):
    r"""Ishigami test function.

    three-dimensional function (usually evaluated on `[-pi, pi]^3`):

        f(x) = sin(x_1) + a sin(x_2)^2 + b x_3^4 sin(x_1)

    Here `a` and `b` are constants where a=7 and b=0.1 or b=0.05
    Proposed to test sensitivity analysis methods because it exhibits strong
    nonlinearity and nonmonotonicity and a peculiar dependence on x_3.
    """

    def __init__(
        self,
        b: float = 0.1,
        noise_std: Optional[float] =None,
        negate: bool = False,
    ) -> None:
        r"""
        Args:
            b: the b constant, should be 0.1 or 0.05.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negative the objective.
        """
        self._optimizers = None
        if b not in (0.1, 0.05):
            raise ValueError("b parameter should be 0.1 or 0.05")
        self.dim =3
        if b == 0.1:
            self.si =[0.3138, 0.4424, 0]
            self.si_t =[0.558, 0.442, 0.244]
            self.s_ij= [0, 0.244, 0]
            self.dgsm_gradient = [0.0005, 0.0005, 0.0005]
            self.dgsm_gradient_abs =[1.88, 4.45, 1.98]
            self.dgsm_gradient_square = [7.7, 24.5, 11]
        elif b == 0.05:
            self.si = [0.218, 0.687, 0]
            self.si_t = [0.3131, 0.6868, 0.095]
            self.s_ij = [0, 0.094, 0]
            self.dgsm_gradient = [0.0004, 0.0004, 0.0004]
            self.dgsm_gradient_abs = [1.26, 4.45, 0.98]
            self.dgsm_gradient_square =[2.8, 24.5, 2.75]
        self._bounds= [(-math.pi, math.pi) for _ in range(self.dim)]
        self.b =b
        super().__init__(noise_std = noise_std, negate=negate)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def compute_dgsm(self, X: Tensor) -> Tuple[List[float], List[float], List[float]]:
        dx_1 = torch.cos(X[..., 0]) * (1 + self.b * (X[..., 2] ** 4))
        dx_2= 14 * torch.cos(X[..., 1]) * torch.sin(X[..., 1])
        dx_3 = self.b * 4 * (X[..., 2] ** 3) * torch.sin(X[..., 0])
        gradient_measure =[
            torch.mean(dx_1).item(),
            torch.mean(dx_1).item(),
            torch.mean(dx_1).item(),
        ]
        gradient_absolute_measure =[
            torch.mean(torch.abs(dx_1)).item(),
            torch.mean(torch.abs(dx_2)).item(),
            torch.mean(torch.abs(dx_3)).item(),
        ]
        gradient_square_measure =[
            torch.mean(torch.pow(dx_1, 2)).item(),
            torch.mean(torch.pow(dx_2, 2)).item(),
            torch.mean(torch.pow(dx_3, 2)).item(),
        ]
        return gradient_measure, gradient_absolute_measure, gradient_square_measure

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device =X.device, dtype=X.dtype)
        t = (
            torch.sin(X[..., 0])
            + 7 * (torch.sin(X[..., 1]) ** 2)
            + self.b * (X[..., 2] ** 4) * torch.sin(X[..., 0])
        )
        return t
    
    def evaluate_true_gradient(self, X: Tensor)-> Tensor:
        self.to(device = X.device, dtype=X.dtype)
        dx_1= torch.cos(X[..., 0]) * (1 + self.b * (X[..., 2] ** 4))
        dx_2 =14 * torch.cos(X[..., 1]) * torch.sin(X[..., 1])
        dx_3 =self.b * 4 * (X[..., 2] ** 3) * torch.sin(X[..., 0])
        gradient= torch.stack([dx_1, dx_2, dx_3], dim=-1)
        return gradient
    

class Lim(SyntheticTestFunction):

    def __init__(
        self,
        noise_std: Optional[float] =None,
        negate: bool = False,
    ) -> None:
        r"""
        Args:
            dim: Dimensionality of the problem. If 6, 8, or 15, will use standard a.
            a: a parameter, unless dim is 6, 8, or 15.
            noise_std: Standard deviation of observation noise.
            negate: Return negatie of function.
        """
        self._optimizers =None
        self.dim = 2
        self._bounds =[(0, 1) for _ in range(self.dim)]
        
        super().__init__(noise_std = noise_std, negate=negate)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device= X.device, dtype=X.dtype)
        x1, x2 = X[..., 0], X[..., 1]
        return ((30 + 5*x1*torch.sin(5*x1)) * (4 + torch.exp(-5*x2)) - 100) / 6


class AnisotropicValley(SyntheticTestFunction):
    r"""
    Two-dimensional valley with a stiff direction along x1.
    """

    def __init__(
        self,
        noise_std: Optional[float]= None,
        negate: bool = False,
        slope: float= 4.0,
        curvature: float= 40.0,
    ) -> None:
        self._optimizers = None
        self.dim = 2
        self.slope = slope
        self.curvature= curvature
        self._bounds = [(-2.0, 2.0), (-1.0, 1.0)]
        super().__init__(noise_std = noise_std, negate=negate)

    @property
    def _optimal_value(self) -> float:
        return 0.0

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device = X.device, dtype=X.dtype)
        x1 = X[..., 0]
        x2 = X[..., 1]
        valley = (x2 - torch.sin(self.slope * x1) / 3.0) ** 2
        narrow = self.curvature * (x1 + 1.2) ** 2
        ripple = 0.2 * torch.sin(6 * x2)
        return valley + narrow + ripple
class Morris(SyntheticTestFunction):
    r"""Morris test function.

    20-dimensional function (usually evaluated on `[0, 1]^20`):

        f(x) = sum_{i=1}\^20 beta_i w_i + sum_{i<j}\^20 beta_ij w_i w_j
        + sum_{i<j<l}\^20 beta_ijl w_i w_j w_l + 5w_1 w_2 w_3 w_4
    """

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool= False,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of observation noise.
            negate: Return negative of function.
        """
        self._optimizers = None
        self.dim = 20
        self._bounds= [(0, 1) for _ in range(self.dim)]
        self.si= [
            0.005,
            0.008,
            0.017,
            0.009,
            0.016,
            0,
            0.069,
            0.1,
            0.15,
            0.1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        self.dgsm_gradient = [
            -47.6840,
            -47.6723,
            -9.2987,
            -47.6605,
            -9.4831,
            -47.6790,
            -9.4401,
            -28.8511,
            -26.1642,
            -28.8462,
            -68.1501,
            -66.8496,
            -68.1661,
            -66.8513,
            -68.1640,
            -66.8495,
            -68.1679,
            -66.8555,
            -68.1426,
            -66.8441,
        ]
        self.dgsm_gradient_abs = [
            135.1225,
            135.1225,
            117.4992,
            135.0901,
            117.5788,
            118.0753,
            77.4511,
            84.4297,
            84.5565,
            84.4296,
            91.7510,
            91.2040,
            91.7662,
            91.2047,
            91.7688,
            91.2029,
            91.7723,
            91.1978,
            91.7529,
            91.1923,
        ]
        self.dgsm_gradient_square = [
            36629.7414,
            36627.3141,
            103987.9600,
            36615.9760,
            103889.2989,
            25468.0737,
            44102.8125,
            13611.9649,
            13501.9957,
            13614.3125,
            17458.1122,
            17251.0162,
            17464.5750,
            17253.2673,
            17464.2558,
            17250.9887,
            17464.3090,
            17252.5585,
            17457.3987,
            17248.9220,
        ]
        super().__init__(noise_std= noise_std, negate=negate)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device =X.device, dtype=X.dtype)
        W= []
        t1 = 0
        t2= 0
        t3 = 0
        for i in range(self.dim):
            if i in [2, 4, 6]:
                wi = 2 * (1.1 * X[..., i] / (X[..., i] + 0.1) - 0.5)
            else:
                wi = 2 * (X[..., i] - 0.5)
            W.append(wi)
            if i < 10:
                betai = 20
            else:
                betai = (-1) ** (i + 1)
            t1 = t1 + betai * wi
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                if i < 6 or j < 6:
                    beta_ij = -15
                else:
                    beta_ij =(-1) ** (i + j + 2)
                t2= t2 + beta_ij * W[i] * W[j]
                for k in range(j + 1, self.dim):
                    if i < 5 or j < 5 or k < 5:
                        beta_ijk= -10
                    else:
                        beta_ijk = 0
                    t3= t3 + beta_ijk * W[i] * W[j] * W[k]
        t4 = 5 * W[0] * W[1] * W[2] * W[3]
        return t1 + t2 + t3 + t4

class Hartmann(SyntheticTestFunction):
    r"""Hartmann synthetic test function.
    H(x) = - sum_{i=1}^4 alpha_i exp( - sum_{j=1}^6 A_ij (x_j - P_ij)**2 )
    """

    def __init__(
        self,
        dim=6,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        if dim not in (4, 6):
            raise ValueError(f"Hartmann with dim {dim} not defined")
        self.dim = dim
        if bounds is None:
            bounds = [(0.0, 1.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)
        self.register_buffer("alpha", torch.tensor([1.0, 1.2, 3.0, 3.2]))
        if dim == 4:
            A = [
                [10, 3, 17, 3.5],
                [0.05, 10, 17, 0.1],
                [3, 3.5, 1.7, 10],
                [17, 8, 0.05, 10],
            ]
            P = [
                [1312, 1696, 5569, 124.0],
                [2329, 4135, 8307, 3736],
                [2348, 1451, 3522, 2883],
                [4047, 8828, 8732, 5743],
            ]
        elif dim == 6:
            A = [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
            P = [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381.0],
            ]
        else:
            raise NotImplementedError
        self.register_buffer("A", torch.tensor(A))
        self.register_buffer("P", torch.tensor(P))

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        inner_sum = torch.sum(
            self.A * (X.unsqueeze(-2) - 0.0001 * self.P).pow(2), dim=-1
        )
        H = -(torch.sum(self.alpha * torch.exp(-inner_sum), dim=-1))
        if self.dim == 4:
            H = (1.1 + H) / 0.839
        if self.dim == 6:
            H = (2.58 + H) / 1.94
        return H

class Gsobol(SyntheticTestFunction):
    r"""Gsobol test function.

    d-dimensional function (usually evaluated on `[0, 1]^d`):

        f(x) = Prod_{i=1}\^{d} ((\|4x_i-2\|+a_i)/(1+a_i)), a_i >=0

    common combinations of dimension and a vector:

        dim=8, a= [0, 1, 4.5, 9, 99, 99, 99, 99]
        dim=6, a=[0, 0.5, 3, 9, 99, 99]
        dim = 15, a= [1, 2, 5, 10, 20, 50, 100, 500, 1000, ..., 1000]
    """

    def __init__(
        self,
        dim: int,
        a: List = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        r"""
        Args:
            dim: Dimensionality of the problem. If 6, 8, or 15, will use standard a.
            a: a parameter, unless dim is 6, 8, or 15.
            noise_std: Standard deviation of observation noise.
            negate: Return negatie of function.
        """
        self._optimizers = None
        self.dim = dim
        self.continuous_inds = list(range(self.dim))
        self.dists = [{"type": "unif", "a": 0., "b": 1.}]*self.dim
        self._bounds = [(0, 1) for _ in range(self.dim)]
        if dim == 6:
            self.a = [0, 0.5, 3, 9, 99, 99]
            self.dgsm_gradient_square = [
                1.8814e01,
                9.7107e00,
                1.5355e00,
                2.4996e-01,
                2.5077e-03,
                2.5077e-03,
            ]

        elif dim == 10:
            self.a = [0, 0, 6.52, 6.52, 6.52, 6.52, 6.52, 6.52, 6.52, 6.52]
            self.dgsm_gradient_square = [
                22.3485,
                22.3561,
                0.5241,
                0.5240,
                0.5241,
                0.5240,
                0.5241,
                0.5241,
                0.5241,
                0.5240,
            ]
        elif dim == 15:
            self.a = [
                1,
                2,
                5,
                10,
                20,
                50,
                100,
                500,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
            ]
            self.dgsm_gradient_square = [
                4.2009e00,
                1.9506e00,
                5.0098e-01,
                1.5003e-01,
                4.1247e-02,
                6.9977e-03,
                1.7844e-03,
                7.2523e-05,
                1.8167e-05,
                1.8167e-05,
                1.8167e-05,
                1.8167e-05,
                1.8167e-05,
                1.8167e-05,
                1.8167e-05,
            ]
        else:
            self.a = a
        self.optimal_sobol_indicies()
        super().__init__(noise_std=noise_std, negate=negate)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def optimal_sobol_indicies(self):
        vi = []
        dim = self.dim
        for i in range(dim):
            vi.append(1 / (3 * ((1 + self.a[i]) ** 2)))
        self.vi = Tensor(vi)
        self.V = torch.prod((1 + self.vi)) - 1
        self.si = self.vi / self.V
        si_t = []
        for i in range(dim):
            si_t.append(
                (
                    self.vi[i]
                    * torch.prod(self.vi[:i] + 1)
                    * torch.prod(self.vi[i + 1 :] + 1)
                )/ self.V)
        self.si_t = Tensor(si_t)

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        t = 1
        dim = self.dim
        for i in range(dim):
            t = t * (torch.abs(4 * X[..., i] - 2) + self.a[i]) / (1 + self.a[i])
        return t        