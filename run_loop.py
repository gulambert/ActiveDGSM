import argparse
import gc
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from botorch.exceptions.warnings import InputDataWarning
from botorch.optim.optimize import optimize_acqf
from gpytorch.utils.warnings import NumericalWarning
from torch import Tensor
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm
from botorch.utils.sampling import draw_sobol_samples

from utils.acquisition import (
    GlobalGradVarRed,
    GlobalGradVarRedKmeans,
    GradVarRed,
    GradMaxVar,
)
from utils.model import GP
from utils.util import MetricEvaluator, get_default_device, select_problem
warnings.filterwarnings("ignore", category = NumericalWarning)
warnings.filterwarnings("ignore", category =RuntimeWarning)
warnings.filterwarnings("ignore", category= InputDataWarning)
warnings.filterwarnings("ignore", category = TqdmExperimentalWarning)

global_methods ={
    "GlobalGradVarRed": GlobalGradVarRed,
    "GlobalGradVarRedKmeans": GlobalGradVarRedKmeans,
}

local_methods = {
    "GradVarRed": GradVarRed,
    "GradMaxVar": GradMaxVar,
}

supported_methods = {**global_methods, **local_methods}

def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def _state_dict_to_cpu(state: Dict[str, Any]) -> Dict[str, Any]:
    cpu_state: Dict[str, Any] ={}
    for key, value in state.items():
        if torch.is_tensor(value):
            cpu_state[key] = value.detach().cpu()
        elif isinstance(value, np.ndarray):
            cpu_state[key] =value.copy()
        else:
            cpu_state[key] =value
    return cpu_state


def _save_model_state(
    state: Dict[str, Any],
    output_dir: Optional[Path],
    iteration: Optional[int],
) -> Optional[str]:
    if output_dir is None:
        return None
    output_dir.mkdir(parents= True, exist_ok=True)
    label = "initial" if iteration is None else f"iter_{iteration + 1:03d}"
    filepath = output_dir / f"model_state_{label}.pt"
    torch.save(state, filepath)
    return str(filepath)


def _init_acquisition(
    method_name: str,
    problem,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    kmeans_config: str = "default",
    global_points_factor: int = 50,
    num_fantasies: int = 8,
) -> Optional[Callable[..., Any]]:
    if method_name in global_methods:
        max_global = 2000 if problem.bounds.shape[1] <= 15 else 1000
        global_points_count =min(problem.bounds.shape[1] * global_points_factor, max_global)
        global_points = draw_sobol_samples(
            n = global_points_count,
            q= 1,
            bounds = problem.bounds,
        ).squeeze(1).to(device =device, dtype=dtype)
        acq_cls = global_methods[method_name]
        base_kwargs ={
            "global_points": global_points,
            "num_fantasies": num_fantasies,
        }
        if method_name == "GlobalGradVarRedKmeans":
            base_kwargs["chunk_kmeans"] = kmeans_config

        def build(*, model):
            return acq_cls(model = model, **base_kwargs)

        return build

    if method_name in local_methods:
        acq_cls =local_methods[method_name]
        base_kwargs: Dict[str, Any] = {}
        if method_name == "GradVarRed":
            base_kwargs["num_fantasies"] = num_fantasies

        def build(*, model):
            return acq_cls(model =model, **base_kwargs)

        return build

    return None


def _format_candidat(cand: Tensor) -> Tensor:
    """
    Ensure every acquisition candidate has the shape (1, d) so that it can be
    concatenated with the design matrix regardless of how it was generated.
    """
    if cand.dim() == 1:
        return cand.unsqueeze(0)
    if cand.dim() == 2:
        if cand.shape[0] != 1:
            raise ValueError("Only a single candidate can be appended at a time.")
        return cand
    return cand.reshape(1, -1)


def run_loop(
    function_name: str,
    method_name: str,
    n_init: int =35,
    num_al_iter: int = 70,
    seed: int = 1000,
    metrics_every: int = 1,
    metrics_sample_size: int =10000,
    grad_q2: bool = False,
    retrain_from_scratch: bool =True,
    no_std: bool = False,
    show_progress: bool =True,
    log_metrics: bool = True,
    kmeans_config: str = "default",
    global_points_factor: int = 50,
    num_fantasies: int = 8,
    state_dict_dir: Optional[str]= None,
) -> Dict[str, Any]:

    allowed_methods =set(supported_methods.keys()) | {"Sobol"}
    if method_name not in allowed_methods:
        raise ValueError(
            "method_name must be one of "
            f"{', '.join(sorted(allowed_methods))}."
        )
    if metrics_every < 1:
        raise ValueError("metrics_every must be >= 1.")

    state_dir= Path(state_dict_dir).expanduser() if state_dict_dir else None

    overall_start = time.time()
    dtype= torch.double
    device =get_default_device()

    problem= select_problem(function_name, device=device, dtype=dtype)
    bounds = problem.bounds.to(device=device, dtype=dtype)
    set_random_seed(291999)
    metric_evaluator = MetricEvaluator(
        problem = problem,
        bounds = bounds,
        device = device,
        dtype = dtype,
        q2_samples =metrics_sample_size,
        sobol_samples = metrics_sample_size,
        compute_grad_q2 =grad_q2,
        sobol_seed= 291999,
    )
    set_random_seed(seed)

    X = (
        draw_sobol_samples(bounds= bounds, n=n_init, q=1)
        .squeeze(1)
        .to(device = device, dtype=dtype)
    )
    Y =problem(X).unsqueeze(-1)

    model = GP(device=device.type, dtype=dtype, no_std_norm=no_std).fit(
        X,
        Y,
        verbose = True,
    )

    acquisition_init =_init_acquisition(method_name, problem, device, dtype, seed, kmeans_config, global_points_factor, num_fantasies)
    sobol_candidates= None
    if method_name == "Sobol":
        sobol_candidates = draw_sobol_samples(
            n = num_al_iter,
            bounds = bounds,
            q= 1
        ).squeeze(1).to(device= device, dtype = dtype)

    initial_design = {
        "X": X.detach().cpu().clone(),
        "Y": Y.detach().cpu().clone(),
    }

    metrics_dict: Dict[str, Any] = {
        "DGSM": [],
        "S1": [],
        "ST": [],
        "opt_time": [],
        "X": [],
        "Y": [],
        "state_dicts": [],
        "state_dict_paths": [],
        "Q2": [],
        "Q2_grad": [],
        "acq_values": [],
        "initial_design": initial_design,
        "added_points": [],
        "total_time": 0.0,
    }

    initial_state =_state_dict_to_cpu(model.state_dict())
    metrics_dict["state_dicts"].append(initial_state)
    metrics_dict["state_dict_paths"].append(
        _save_model_state(initial_state, state_dir, iteration =None)
    )

    initial_metrics = metric_evaluator.evaluate(model)
    metrics_dict["Q2"].append(initial_metrics["Q2"])
    metrics_dict["DGSM"].append(initial_metrics["DGSM"])
    metrics_dict["S1"].append(initial_metrics["S1"])
    metrics_dict["ST"].append(initial_metrics["ST"])
    metrics_dict["Q2_grad"].append(initial_metrics["Q2_grad"])
    if log_metrics:
        print("Q2 Score (init) : {:.4f}".format(initial_metrics["Q2"]))
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(range(num_al_iter), desc="AL step", dynamic_ncols=True)
        iteration_iterable =progress_bar
    else:
        iteration_iterable= range(num_al_iter)

    for iteration in iteration_iterable:
        start =time.time()
        acq_value: Optional[float]= None
        if acquisition_init is not None:
            acqf = acquisition_init(model=model)
            cand, acqf_values = optimize_acqf(
                acq_function = acqf,
                bounds= bounds,
                q = 1,
                num_restarts = 20,
                raw_samples = 256,
                options= {"batch_limit": 10},    # you may opt out and change the batch limit to alievate memory/perfomance issues
            )
            cand = _format_candidat(cand).to(device=device, dtype=dtype)
            acq_value= float(acqf_values.squeeze().detach().cpu().item())
        else:
            if sobol_candidates is None:
                raise RuntimeError("Sobol' candidates not computed.")
            cand = _format_candidat(sobol_candidates[iteration]).to(
                device = device, dtype=dtype
            )

        opt_time= time.time() - start
        metrics_dict["opt_time"].append(opt_time)
        metrics_dict["acq_values"].append(acq_value)

        new_y = problem(cand).to(device=device, dtype=dtype).unsqueeze(-1)
        cand_cpu = cand.detach().cpu()
        new_y_cpu =new_y.detach().cpu()
        metrics_dict["added_points"].append(
            {
                "X": cand_cpu.squeeze(0),
                "Y": new_y_cpu.squeeze(0),
                "acq_value": acq_value,
                "iteration_time": opt_time,
            }
        )
        X, Y = torch.cat((X, cand), dim=0), torch.cat((Y, new_y), dim=0)

        if retrain_from_scratch:
            model= GP(device=device.type, dtype=dtype, no_std_norm=no_std).fit(
                X,
                Y,
                verbose = False,
            )
        else:
            prev_state = model.state_dict()
            model.fit(X, Y, verbose = False, state_dict=prev_state)
        state_snapshot = _state_dict_to_cpu(model.state_dict())
        metrics_dict["state_dicts"].append(state_snapshot)
        metrics_dict["state_dict_paths"].append(
            _save_model_state(state_snapshot, state_dir, iteration = iteration)
        )

        metrics= metric_evaluator.evaluate(model)
        metrics_dict["Q2"].append(metrics["Q2"])
        metrics_dict["DGSM"].append(metrics["DGSM"])
        metrics_dict["S1"].append(metrics["S1"])
        metrics_dict["ST"].append(metrics["ST"])
        metrics_dict["Q2_grad"].append(metrics["Q2_grad"])
        if log_metrics and (iteration + 1) % metrics_every == 0:
            print(metrics["Q2"])
            if show_progress and progress_bar is not None:
                progress_bar.set_postfix(q2= metrics["Q2"], time=f"{opt_time:.2f}s")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_dict["X"].append(X.detach().cpu())
    metrics_dict["Y"].append(Y.detach().cpu())

    if show_progress and progress_bar is not None:
        progress_bar.close()

    metrics_dict["total_time"]= time.time() - overall_start
    return metrics_dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Active learning loop runner.")
    parser.add_argument(
        "--function",
        type = str,
        default= "Ishigami2",
        help= "Name of the synthetic test function to use (see utils.util.available_problems).",
    )
    parser.add_argument(
        "--method",
        type =str,
        choices = [
            "GlobalGradVarRed",
            "GlobalGradVarRedKmeans",
            "GradVarRed",
            "GradMaxVar",
            "Sobol",
        ],
        default = "GlobalGradVarRed",
    )
    parser.add_argument("--n-init", type= int, default=20)
    parser.add_argument("--num-al-iter", type =int, default=280)
    parser.add_argument("--seed", type = int, default=1000)
    parser.add_argument("--metrics-every", type= int, default=10)
    parser.add_argument("--metrics-sample-size", type = int, default=128)
    parser.add_argument(
        "--grad-q2",
        action= "store_true",
        help = "Also track Q2 on partial derivatives when a true gradient is available.",
    )
    parser.add_argument(
        "--kmeans-config",
        type=str,
        choices=["default", "balanced"],
        default="default",
        help="Chunk-selection strategy for GlobalGradVarRedKmeans: 'default' uses "
        "regular k-means (uneven cluster sizes), 'balanced' uses constrained "
        "k-means (equal-size clusters). Ignored by other methods.",
    )
    parser.add_argument(
        "--global-points-factor",
        type=int,
        default=50,
        help="Number (N in the article) of global points Xs per input dimension for "
        "GlobalGradVarRed/GlobalGradVarRedKmeans (points = min(dim * factor, "
        "2000 if dim <= 15 else 1000)). Ignored by other methods.",
    )
    parser.add_argument(
        "--num-fantasies",
        type=int,
        default=8,
        help="Number of fantasy samples used by the look-ahead reduction in "
        "GradVarRed, GlobalGradVarRed, and GlobalGradVarRedKmeans. "
        "Ignored by GradMaxVar and Sobol.",
    )
    parser.add_argument(
        "--retrain-from-scratch",
        action = "store_true",
        help = "Fully retrain the GP at every iteration instead of warm-starting.",
    )
    parser.add_argument(
        "--no-std",
        action = "store_true",
        help = "Disable GP output normalisation (no_std_norm).",
    )
    parser.add_argument(
        "--state-dict-dir",
        type = str,
        default = None,
        help = "Directory to save model state dicts to.",
    )
    return parser.parse_args()


def main() -> None:
    args= _parse_args()
    metrics =run_loop(
        function_name = args.function,
        method_name = args.method,
        n_init = args.n_init,
        num_al_iter = args.num_al_iter,
        seed = args.seed,
        metrics_every= args.metrics_every,
        metrics_sample_size =args.metrics_sample_size,
        grad_q2 = args.grad_q2,
        retrain_from_scratch =args.retrain_from_scratch,
        no_std =args.no_std,
        kmeans_config=args.kmeans_config,
        global_points_factor=args.global_points_factor,
        num_fantasies=args.num_fantasies,
        state_dict_dir = args.state_dict_dir,
    )
    final_q2 = next((q for q in reversed(metrics["Q2"]) if q is not None), None)
    print(f"Finished. Final Q2: {final_q2}")

if __name__ == "__main__":
    main()
