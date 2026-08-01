import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.utils.sampling import draw_sobol_samples
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_samp
from torch import Tensor
from torchmetrics.functional.regression import r2_score
from utils.functions import Ishigami, Lim, AnisotropicValley,Morris, Hartmann, Gsobol

ProblemBuilder =Callable[..., SyntheticTestFunction]


def get_default_device() -> torch.device:
    """Select CUDA (float64) when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _power_of_two(value: int, *, min_exp: int =7, max_exp: int = 16) -> int:
    """Clamp `value` to the nearest power of two within provided bounds."""
    if value <= 0:
        return 2 ** min_exp
    exponent = int(math.log2(value))
    exponent= max(min_exp, min(exponent, max_exp))
    return 2 ** exponent


class _GroupArray(np.ndarray):
    """np.ndarray subclass with a well-defined truth value for SALib checks."""
    def __new__(cls, values: np.ndarray):
        obj = np.asarray(values, dtype=int).view(cls)
        return obj

    def __bool__(self) -> bool:
        return bool(self.size)


def _build_sobol_problem(bounds: Tensor) -> Dict[str, Any]:
    """Create a SALib-compatible problem dictionary from BoTorch bounds."""
    dim = bounds.shape[1]
    return {
        "num_vars": dim,
        "names": [f"x{i+1}" for i in range(dim)],
        "bounds": bounds.T.detach().cpu().tolist(),
        "groups": _GroupArray(np.arange(dim, dtype =int)),
    }


class MetricEvaluator:
    """Compute Q2, DGSM, Sobol, and optional gradient-Q2 metrics."""
    def __init__(
        self,
        *,
        problem: SyntheticTestFunction,
        bounds: Tensor,
        device: torch.device,
        dtype: torch.dtype,
        q2_samples: int,
        sobol_samples: Optional[int] = None,
        sobol_batch_size: int = 4096,
        compute_grad_q2: bool =False,
        sobol_seed: Optional[int] = None,
    ) -> None:
        self.problem= problem
        self.bounds = bounds
        self.device = device
        self.dtype = dtype
        self.dim= bounds.shape[1]
        self.q2_samples =max(q2_samples, 128)
        self.test_X = draw_sobol_samples(
            bounds =bounds, n=self.q2_samples, q=1
        ).squeeze(1).to(device = device, dtype=dtype)
        with torch.no_grad():
            self.test_Y = problem(self.test_X).unsqueeze(-1)

        self.grad_available= bool(
            compute_grad_q2 and hasattr(problem, "evaluate_true_gradient")
        )
        self.test_grad: Optional[Tensor] = None
        if self.grad_available:
            with torch.no_grad():
                grad_fn = getattr(problem, "evaluate_true_gradient")
                self.test_grad = grad_fn(self.test_X)

        sobol_base = sobol_samples if sobol_samples is not None else self.q2_samples
        self.sobol_N= _power_of_two(sobol_base)
        self.sobol_problem= _build_sobol_problem(bounds)
        self.sobol_batch_size = max(512, min(sobol_batch_size, 8192))
        saltelli_kwargs = {"calc_second_order": False}
        if sobol_seed is not None:
            saltelli_kwargs["seed"] = sobol_seed
        self._sobol_samples= sobol_samp.sample(
            self.sobol_problem,
            self.sobol_N,
            **saltelli_kwargs,
        ).astype(np.float64)

    def _batched_model_mean(self, model, samples: np.ndarray) -> np.ndarray:
        outputs: List[Tensor] = []
        total = samples.shape[0]
        for start in range(0, total, self.sobol_batch_size):
            chunk_np= samples[start : start + self.sobol_batch_size]
            chunk = torch.from_numpy(chunk_np).to(device=self.device, dtype=self.dtype)
            with torch.no_grad():
                post = model.predict(chunk, return_posterior=True)
                mean= post.mean.squeeze(-1)
            outputs.append(mean.detach().cpu())
        return torch.cat(outputs, dim = 0).numpy()

    def _compute_q2(self, model) -> float:
        with torch.no_grad():
            post= model.predict(self.test_X, return_posterior=True)
            mean = post.mean.unsqueeze(-1)
        return float(r2_score(mean, self.test_Y))

    def _compute_grad_q2(self, model) -> Optional[Tensor]:
        if not self.grad_available or self.test_grad is None:
            return None
        grads = model.grad_posterior(self.test_X, only_mean=True)
        grad_q2: List[float] = []
        for dim in range(self.dim):
            pred =grads[:, dim].unsqueeze(-1).detach()
            target = self.test_grad[:, dim].unsqueeze(-1).detach()
            grad_q2.append(float(r2_score(pred, target)))
        return torch.tensor(grad_q2, dtype = torch.float64)

    def _compute_dgsm(self, model) -> Tensor:
        grads= model.grad_posterior(self.test_X, only_mean=True)
        return grads.pow(2).mean(dim = 0).detach().cpu()

    def _compute_sobol(self, model) -> Tuple[Tensor, Tensor]:
        y_vals = self._batched_model_mean(model, self._sobol_samples)
        sob = sobol.analyze(
            self.sobol_problem,
            Y = y_vals,
            calc_second_order= False,
            print_to_console = False,
        )
        s1_vals = np.nan_to_num(sob["S1"], nan=0.0, posinf=0.0, neginf=0.0)
        st_vals = np.nan_to_num(sob["ST"], nan=0.0, posinf=0.0, neginf=0.0)
        s1 = torch.from_numpy(s1_vals).to(dtype=torch.float64)
        st =torch.from_numpy(st_vals).to(dtype=torch.float64)
        return s1, st

    def evaluate(self, model) -> Dict[str, Any]:
        metrics: Dict[str, Any]= {}
        metrics["Q2"] =self._compute_q2(model)
        metrics["DGSM"] = self._compute_dgsm(model)
        metrics["S1"], metrics["ST"] = self._compute_sobol(model)
        metrics["Q2_grad"]= self._compute_grad_q2(model)
        return metrics


_custom_problems: Dict[str, ProblemBuilder] = {
    "ishigami": partial(Ishigami, b = 0.05),
    "ishigami2": partial(Ishigami, b =0.05),
    "lim": Lim,
    "anisotropicvalley": AnisotropicValley,
    "valley": AnisotropicValley,
    "morris": Morris,
    "hartmann4": partial(Hartmann, dim = 4),
     "hartmann6": partial(Hartmann, dim = 6),
     "gsobol6" : partial(Gsobol, dim=6),
     "gsobol10": partial(Gsobol, dim=10),
     "gsobol15": partial(Gsobol, dim=15)
}


def available_problems() -> List[str]:
    """Return alphabetised list of function names that can be selected."""
    return sorted(_custom_problems.keys())


def select_problem(
    function_name: str,
    *,
    function_kwargs: Optional[Dict[str, Any]]= None,
    **to_kwargs: Any,
) -> SyntheticTestFunction:
    if not function_name:
        raise ValueError("function_name must be a non-empty string.")
    key = function_name.lower()
    builder: Optional[ProblemBuilder] = _custom_problems.get(key)
    if builder is None:
        options= ", ".join(available_problems())
        raise ValueError(
            f"Unknown function '{function_name}'. Available options are: {options}"
        )
    problem = builder(**(function_kwargs or {}))
    if to_kwargs:
        problem = problem.to(**to_kwargs)
    return problem
