import numpy as np
import torch

from typing import Optional, List, Dict, Union, Any
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from utils.posterior import (
    batched_posterior_derivative_joint_fantasize,
    batched_posterior_derivative_joint_fantasize_chunked_kmeans,
)
from gpytorch.distributions import MultivariateNormal
from botorch.sampling.normal import SobolQMCNormalSampler

ArrayOrTensor =Union[np.ndarray, torch.Tensor]
torch.set_default_dtype(torch.float64)


class GP:
    """
    Thin wrapper around a BoTorch SingleTaskGP for scalar-output regression.

    Wraps a `SingleTaskGP` (Matern-5/2 ARD kernel, constant mean, fixed small
    observation noise) fitted by exact marginal likelihood maximization, and adds:
      - optional (but highly recommanedd !!) min-max input normalization / output standardization, transparently
        undone on prediction (see `no_std_norm`)
      - an exact gradient posterior GP, `grad_posterior`, obtained by
        differentiating the Matern-5/2 kernel, with an optional k-means-chunked path for large reference sets
      - `fantasize`, returning a new GP conditioned on a hypothetical observation,
        used by the look-ahead acquisition functions in `utils.acquisition`
      - `state_dict` / `load_state_dict`, to checkpoint and restore a fit without
        refitting hyperparameters from scratch.

    Only scalar outputs (output_dim == 1) are supported for now.

    Args:
        device: torch.device
        dtype: torch.float64 (float32 isn't recommanded as Gaussian process covariance inversion maybe ill-conditioned)
        no_std_norm: If True, skip input/output normalization and fit directly on
            the raw data. 
            
            If False (default), inputs are min-max scaled to [0, 1]
            and outputs are standardized to zero mean / unit variance before
            fitting, and every prediction is mapped back to the original scale.
    """

    def __init__(
        self,
        device: str ="cpu",
        dtype: torch.dtype = torch.float64,
        no_std_norm: bool =False,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.no_std_norm = no_std_norm

        if not self.no_std_norm:
            self._x_normalizer = MinMaxNormalizer(device=device, dtype=dtype)
            self._y_standardizer = Standardizer(device=device, dtype=dtype)
        else:
            self._x_normalizer = None
            self._y_standardizer = None

        self.model: Optional[SingleTaskGP] = None
        self.X_train: Optional[torch.Tensor] = None
        self.Y_train: Optional[torch.Tensor] = None
        self.input_dim: Optional[int] = None
        self.output_dim: Optional[int]= None

    def _ensure_tensor(self, X: ArrayOrTensor) -> torch.Tensor:
        if isinstance(X, np.ndarray):
            return torch.as_tensor(X, dtype = self.dtype, device=self.device)
        if isinstance(X, torch.Tensor):
            return X.to(dtype= self.dtype, device=self.device)
        raise TypeError("Unsupported input type. Expected numpy or torch tensor")

    def _unstandardize_mvn(self, mvn_std: MultivariateNormal) -> MultivariateNormal:
        """
        Take a MultivariateNormal in standardized output space and map it back
        to the original output scale using the stored Standardizer.
        Assumes scalar output (output_dim == 1), which your GP.fit enforces.
        """
        if self.no_std_norm:
            return mvn_std

        if (
            self._y_standardizer is None
            or self._y_standardizer.mean_ is None
            or self._y_standardizer.std_ is None
        ):
            raise RuntimeError("Standardizer parameters are missing,  cannot unstandardize posterior.")

        # scalar σ, μ (since output_dim == 1)
        std = self._y_standardizer.std_.view(-1)[0]
        mean =self._y_standardizer.mean_.view(-1)[0]

        std =std.to(device=mvn_std.mean.device, dtype=mvn_std.mean.dtype)
        mean =mean.to(device=mvn_std.mean.device, dtype=mvn_std.mean.dtype)

        # Affine transform: Y = σ * Y_std + μ
        mean_orig= mvn_std.mean * std + mean
        cov_orig = mvn_std.covariance_matrix * (std ** 2)

        return MultivariateNormal(mean_orig, cov_orig)

    def fit(
        self,
        x_array: ArrayOrTensor,
        y_array: ArrayOrTensor,
        verbose: bool = True,
        state_dict: Optional[Dict[str, Any]] = None,
    ):
        x_tensor = self._ensure_tensor(x_array)
        y_tensor = self._ensure_tensor(y_array)

        if y_tensor.dim() == 1:
            y_tensor= y_tensor.unsqueeze(-1)
        if x_tensor.dim() != 2 or y_tensor.dim() != 2:
            raise ValueError("x_array and y_array must be 2D after processing")
        if x_tensor.shape[0] != y_tensor.shape[0]:
            raise ValueError("x_array and y_array must share the same number of samples")

        self.input_dim = x_tensor.shape[1]
        self.output_dim = y_tensor.shape[1]
        if self.output_dim != 1:
            raise ValueError(f"GP supports scalar outputs only, received output_dim={self.output_dim}")

        if state_dict is not None:
            self.load_state_dict(state_dict)

            if self.no_std_norm:
                X_norm, Y_norm = x_tensor, y_tensor
            else:
                if self._x_normalizer is None or self._y_standardizer is None:
                    raise RuntimeError("Normalizers are missing when restoring from state_dict.")
                X_norm = self._x_normalizer.transform(x_tensor)
                Y_norm = self._y_standardizer.transform(y_tensor)

            self.X_train, self.Y_train = X_norm, Y_norm
            restored =self._fit_model(X_norm, Y_norm, model_state=state_dict.get("model_state"))

            if verbose:
                origin = "restored" if restored else "refit"
                print(
                    f"[GP.fit] {origin} from state_dict (no_std_norm={self.no_std_norm})"
                )
            return self

        if self.no_std_norm:
            X_norm, Y_norm = x_tensor, y_tensor
        else:
            self._x_normalizer.fit(x_tensor)
            self._y_standardizer.fit(y_tensor)
            X_norm = self._x_normalizer.transform(x_tensor)
            Y_norm = self._y_standardizer.transform(y_tensor)

        self.X_train, self.Y_train= X_norm, Y_norm
        self._fit_model(X_norm, Y_norm)

        if verbose:
            print(
                f"(Fit) trained from scratch: n={x_tensor.shape[0]}, "
                f"no_std_norm={self.no_std_norm}"
            )
        return self

    def _fit_model(
        self,
        X_norm: torch.Tensor,
        Y_norm: torch.Tensor,
        model_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        gp = SingleTaskGP(
            train_X =X_norm,
            train_Y= Y_norm,
            outcome_transform = None,
            train_Yvar =1e-5 * torch.ones_like(Y_norm),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

        loaded = False
        if model_state is not None:
            try:
                gp.load_state_dict(model_state)
                loaded =True
            except Exception as exc:
                print(f"(Fit) Warning: model state failed: {exc}. Refitting.")

        if not loaded:
            fit_gpytorch_mll(mll)

        self.model = gp
        return loaded

    def predict(
        self,
        x_new: ArrayOrTensor,
        return_torch: bool = True,
        return_posterior: bool = False,
    ):
        X_new = self._ensure_tensor(x_new)

        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_new_norm = X_new if self.no_std_norm else self._x_normalizer.transform(X_new)

        posterior = self.model.posterior(X_new_norm)
        y_mean_std = posterior.mean
        y_var_std = posterior.variance

        mvn_std =posterior.mvn  # in standardized space
        mvn = self._unstandardize_mvn(mvn_std)  # in original space

        if return_posterior:
            return mvn

        if self.no_std_norm:
            y_mean= y_mean_std
            y_var =y_var_std
        else:
            std = self._y_standardizer.std_.view(1, -1)
            mean = self._y_standardizer.mean_.view(1, -1)
            y_mean = y_mean_std * std + mean
            y_var = y_var_std * (std ** 2)

        out = {
            "y_mean_std": y_mean_std,
            "y_var_std": y_var_std,
            "y_mean": y_mean,
            "y_var": y_var,
            "posterior_mvn_std": mvn_std,
            "posterior_mvn": mvn,
        }

        if not return_torch:
            out = {
                key: (val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else val)
                for key, val in out.items()
            }

        return out

    def grad_posterior(
        self,
        X: ArrayOrTensor,
        only_mean: bool = False,
        *,
        chunked: bool = False,
        chunk_size: Optional[int] =None,
        chunk_kmeans: str ="default",
    ):
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if self.output_dim and self.output_dim != 1:
            raise NotImplementedError("grad_posterior currently supports scalar outputs only.")
        if chunked and only_mean:
            raise ValueError("Chunked gradient posterior requires only_mean=False.")

        X = self._ensure_tensor(X)
        if X.dim() != 2:
            raise ValueError("X must be a 2D array/tensor")

        if self.no_std_norm:
            Xt =X
            inv_rng= torch.ones(X.shape[1], device=self.device, dtype=self.dtype)
            std_y = torch.ones(1, device=self.device, dtype=self.dtype)
        else:
            Xt= self._x_normalizer.transform(X)
            inv_rng = 1.0 / self._x_normalizer._rng.squeeze()
            std_y= self._y_standardizer.std_.squeeze()
            if std_y.dim() == 0:
                std_y= std_y.unsqueeze(0)
            std_y =std_y.to(device=self.device, dtype=self.dtype)
            inv_rng= inv_rng.to(device=self.device, dtype=self.dtype)

        N, d =X.shape
        if inv_rng.dim() == 0:
            inv_rng = inv_rng.unsqueeze(0)

        scale_per_dim = (std_y.view(-1)[0] * inv_rng).to(device=self.device, dtype=self.dtype)

        if chunked:
            selection_alg = {
                "default": "kmeans",
                "balanced": "kmeans-equal",
            }.get(chunk_kmeans.lower() if isinstance(chunk_kmeans, str) else chunk_kmeans)
            if selection_alg is None:
                raise ValueError(
                    "chunk_kmeans must be one of {'default','balanced'}."
                )

            chunk_results_std = batched_posterior_derivative_joint_fantasize_chunked_kmeans(
                self.model,
                Xt,
                chunk_size = chunk_size,
                selection_alg = selection_alg,
            )
            scaled_chunks= []
            for mean_std, cov_std in chunk_results_std:
                chunk_entries = mean_std.shape[-1]
                chunk_points = chunk_entries // d
                chunk_scale = scale_per_dim.repeat(chunk_points)
                chunk_scale = chunk_scale.view(1, 1, chunk_points * d)
                mean_orig = mean_std * chunk_scale
                cov_scale = chunk_scale.unsqueeze(-1) * chunk_scale.unsqueeze(-2)
                cov_orig= cov_std * cov_scale
                scaled_chunks.append((mean_orig, cov_orig))
            return scaled_chunks

        if only_mean:
            mean_std = batched_posterior_derivative_joint_fantasize(
                self.model, Xt, return_full = False
            )
            cov_std = None
        else:
            mvn_norm= batched_posterior_derivative_joint_fantasize(self.model, Xt)
            mean_std, cov_std =mvn_norm.mean, mvn_norm.covariance_matrix

        scale_repeated = scale_per_dim.repeat(N)
        scale_repeated =scale_repeated.to(device=self.device, dtype=self.dtype)

        mean_scale = scale_repeated.view(1, 1, N * d)
        mean_orig = mean_std * mean_scale

        if only_mean:
            return mean_orig.view(N, d)

        cov_scale = mean_scale.unsqueeze(-1) * mean_scale.unsqueeze(-2)
        cov_orig= cov_std * cov_scale

        grad_posteriors: List[MultivariateNormal] = [
            MultivariateNormal(mean_orig, covariance_matrix = cov_orig)
        ]
        return grad_posteriors

    def fantasize(
        self,
        Xcond: ArrayOrTensor,
        n_fantasies: int = 8,
        sampler: Optional["SobolQMCNormalSampler"]= None,
    ) -> "GP":
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([n_fantasies]))

        Xcond = self._ensure_tensor(Xcond)
        Xcond_norm = Xcond if self.no_std_norm else self._x_normalizer.transform(Xcond)

        fant_gp = GP(
            device =self.device,
            dtype = self.dtype,
            no_std_norm = self.no_std_norm,
        )

        fant_gp._x_normalizer = self._x_normalizer
        fant_gp._y_standardizer = self._y_standardizer
        fant_gp.X_train = self.X_train
        fant_gp.Y_train = self.Y_train
        fant_gp.input_dim = self.input_dim
        fant_gp.output_dim = self.output_dim

        fant_gp.model = self.model.fantasize(X=Xcond_norm, sampler=sampler)

        return fant_gp

    def state_dict(self) -> Dict[str, Any]:
        state= {
            "no_std_norm": self.no_std_norm,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }
        if self.model is not None:
            state["model_state"] = self.model.state_dict()

        if not self.no_std_norm:
            state.update(
                {
                    "x_min": self._x_normalizer.min_.detach().cpu().numpy()
                    if self._x_normalizer and self._x_normalizer.min_ is not None
                    else None,
                    "x_max": self._x_normalizer.max_.detach().cpu().numpy()
                    if self._x_normalizer and self._x_normalizer.max_ is not None
                    else None,
                    "y_mean": self._y_standardizer.mean_.detach().cpu().numpy()
                    if self._y_standardizer and self._y_standardizer.mean_ is not None
                    else None,
                    "y_std": self._y_standardizer.std_.detach().cpu().numpy()
                    if self._y_standardizer and self._y_standardizer.std_ is not None
                    else None,
                }
            )
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        if state is None:
            return self

        self.input_dim= state.get("input_dim", self.input_dim)
        self.output_dim = state.get("output_dim", self.output_dim)
        self.no_std_norm =state.get("no_std_norm", self.no_std_norm)

        if not self.no_std_norm:
            if self._x_normalizer is None:
                self._x_normalizer = MinMaxNormalizer(device=self.device, dtype=self.dtype)
            if self._y_standardizer is None:
                self._y_standardizer = Standardizer(device=self.device, dtype=self.dtype)

            if state.get("x_min") is not None and state.get("x_max") is not None:
                self._x_normalizer.min_ =torch.from_numpy(state["x_min"]).to(self.device, self.dtype)
                self._x_normalizer.max_ = torch.from_numpy(state["x_max"]).to(self.device, self.dtype)
                self._x_normalizer._rng = self._x_normalizer.max_ - self._x_normalizer.min_

            if state.get("y_mean") is not None and state.get("y_std") is not None:
                self._y_standardizer.mean_ = torch.from_numpy(state["y_mean"]).to(self.device, self.dtype)
                self._y_standardizer.std_ = torch.from_numpy(state["y_std"]).to(self.device, self.dtype)

        return self


class MinMaxNormalizer:
    """Min-max normalizer using torch tensors: X -> (X - min)/(max-min)"""
    def __init__(self, device: Union[str, torch.device] = 'cpu', dtype: torch.dtype = torch.float64):
        self.device= torch.device(device)
        self.dtype =dtype
        self.min_: Optional[torch.Tensor] = None
        self.max_: Optional[torch.Tensor]= None
        self._rng: Optional[torch.Tensor] = None

    def _ensure_tensor(self, X: ArrayOrTensor) -> torch.Tensor:
        """Convert input to torch tensor with correct device and dtype."""
        if isinstance(X, np.ndarray):
            return torch.as_tensor(X, dtype = self.dtype, device=self.device)
        elif isinstance(X, torch.Tensor):
            return X.to(dtype = self.dtype, device=self.device)
        else:
            raise TypeError("Unsupported input type. Expected numpy or torch tensor")

    def fit(self, X: ArrayOrTensor):
        """Fit the normalizer on input data."""
        X = self._ensure_tensor(X)
        self.min_ = X.min(dim=0, keepdim=True)[0]
        self.max_ =X.max(dim=0, keepdim=True)[0]
        rng =(self.max_ - self.min_)
        rng[rng == 0] = 1.0
        self._rng =rng
        return self

    def transform(self, X: ArrayOrTensor) -> torch.Tensor:
        """Transform data using fitted normalizer."""
        X= self._ensure_tensor(X)
        return (X - self.min_) / self._rng

    def transform_pca(self, X: ArrayOrTensor, M: int) -> torch.Tensor:
        """Transform data using last M dimensions of fitted normalizer."""
        X = self._ensure_tensor(X)
        return (X - self.min_[:, -M:]) / self._rng[:, -M:]

    def inverse_transform(self, Xn: ArrayOrTensor) -> torch.Tensor:
        """Inverse transform normalized data back to original scale."""
        Xn= self._ensure_tensor(Xn)
        return Xn * self._rng + self.min_


class Standardizer:
    """Per-dimension z-score: Y -> (Y - mean)/std, using torch tensors."""
    def __init__(self, eps: float = 1e-12, device: Union[str, torch.device] = 'cpu', dtype: torch.dtype = torch.float64):
        self.device = torch.device(device)
        self.dtype =dtype
        self.mean_: Optional[torch.Tensor] = None
        self.std_: Optional[torch.Tensor] = None
        self.eps =eps

    def _ensure_tensor(self, Y: ArrayOrTensor) -> torch.Tensor:
        """Convert input to torch tensor with correct device and dtype."""
        if isinstance(Y, np.ndarray):
            return torch.as_tensor(Y, dtype= self.dtype, device=self.device)
        elif isinstance(Y, torch.Tensor):
            return Y.to(dtype =self.dtype, device=self.device)
        else:
            raise TypeError("Unsupported input type.Expected numpy or torch tensor")

    def fit(self, Y: ArrayOrTensor):
        """Fit the standardizer on input data."""
        Y = self._ensure_tensor(Y)
        self.mean_ =Y.mean(dim=0, keepdim=True)
        std =Y.std(dim=0, keepdim=True, unbiased=False)
        std[std < self.eps] =1.0
        self.std_ = std
        return self

    def transform(self, Y: ArrayOrTensor) -> torch.Tensor:
        """Transform data using fitted standardizer."""
        Y =self._ensure_tensor(Y)
        return (Y - self.mean_) / self.std_

    def inverse_transform(self, Yn: ArrayOrTensor) -> torch.Tensor:
        """Inverse transform standardized data back to original scale."""
        Yn = self._ensure_tensor(Yn)
        return Yn * self.std_ + self.mean_