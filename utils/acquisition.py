from typing import Optional

import torch
from botorch import settings
from botorch.acquisition import AcquisitionFunction
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor
from gpytorch.distributions import MultivariateNormal


def _global_variance(posterior: MultivariateNormal) -> torch.Tensor:
    r"""
    Computes the global variance of the squared norm defined as: Var = 2 * tr(S^2) + 4 * m^T S m,

    where S is the posterior covariance and m is the posterior mean.

    The input posterior is assumed to have:
      - mean of shape (n_fant, B, N*d)
      - covariance_matrix of shape (n_fant, B, N*d, N*d)

    The covariance S is the same across fantasies, the mean m depends
    on the fantasy (i.e observations y). --> we use S from the first fantasy and compute
    m^T S m for each fantasy, then average over fantasies.

    Args:
        posterior: MultivariateNormal with the shapes described above.

    Returns:
        A tensor of shape (B,) containing the global variance for each candidate.
    """
    S= posterior.covariance_matrix[0] # S has shape (B, N*d, N*d)
    trace_S2= torch.sum(S * S, dim=(-2, -1))  # (B,), sum (elementwise^2) over the last 2 dims.
    m= posterior.mean  # (n_fant, B, N*d)
    mSm = torch.einsum('fbi,bij,fbj->fb', m, S, m)  #(n_fant, B), quad form \forall fant f and cand  b, val= m[f, b, :]^T @ S[b] @ m[f, b, :].
    mSm_mean= mSm.mean(dim=0)   #expectation over the fantasies i.e E_y[Var] in the reduction term.
    global_var = 2.0 * trace_S2 + 4.0 * mSm_mean   # (B,)
    return global_var


class GradMaxVar(AcquisitionFunction):
    """Maximise the variance of || \nabla \eta(x)||^2 at the candidate point."""

    def __init__(self, model):
        super().__init__(model = model)

    @t_batch_mode_transform(expected_q = 1)
    def forward(self, X: Tensor) -> Tensor:
        Xs = X.squeeze(1)
        posterior = self.model.grad_posterior(Xs)[0]
        return _global_variance(posterior)


class GradVarRed(AcquisitionFunction):
    """Variance reduction analogue of GlobalGradVarRed, evaluated at the candidate point only."""

    def __init__(self, model, num_fantasies: int = 16):
        super().__init__(model =model)
        self.num_fantasies =num_fantasies
        self.sampler= SobolQMCNormalSampler(torch.Size([num_fantasies]))

    @t_batch_mode_transform(expected_q = 1)
    def forward(self, X: Tensor) -> Tensor:
        Xs =X.squeeze(1)
        posterior_curr = self.model.grad_posterior(Xs)[0]
        curr_var = _global_variance(posterior_curr)
        with settings.propagate_grads(True):
            fantasy_model = self.model.fantasize(Xcond=X, sampler=self.sampler)
            posterior_fant =fantasy_model.grad_posterior(Xs)[0]
        lookahead_var = _global_variance(posterior_fant)
        return curr_var - lookahead_var


class GlobalGradVarRed(AcquisitionFunction):
    def __init__(
        self,
        model,
        global_points: Tensor,    # (N, d)
        num_fantasies: int = 16,
    ):
        super().__init__(model = model)
        self.global_points = global_points
        self.num_fantasies =num_fantasies
        self.sampler= SobolQMCNormalSampler(sample_shape=torch.Size([self.num_fantasies]))
        with torch.no_grad():
            posterior_joint = self.model.grad_posterior(self.global_points)[0]
            self.curr_var = _global_variance(posterior_joint)

    def forward(self, X:Tensor)-> Tensor:
        """
        X : (b, 1, d)

        """
        with settings.propagate_grads(True):
            fantasy_model =self.model.fantasize(Xcond=X, sampler=self.sampler)  # X (b, q, d) allows independent conditionning for the fantasy model batch shape (n_fant, b)
            posterior_fant =fantasy_model.grad_posterior(self.global_points)[0]
        lookahead_var = _global_variance(posterior_fant)
        acqf_val = self.curr_var - lookahead_var   #(b,)
        return acqf_val


def _global_variance_chunked(chunks_list):
    """
    Computes the global variance from a list of chunks with raw tensors.

    Args:
        chunks_list: A list of tuples (mean, cov) for chunks of global points

    Returns:
        A tensor of shape (B,) containing the global variance for each candidate
    """
    n_chunks = len(chunks_list)
    if n_chunks == 0:
        return torch.tensor(0.0, device = chunks_list[0][0].device)

    B = chunks_list[0][0].shape[1]

    trace_S2_total = torch.zeros(B, device=chunks_list[0][0].device)
    mSm_total= torch.zeros((chunks_list[0][0].shape[0], B),
                           device =chunks_list[0][0].device)

    for mean, cov in chunks_list:
        
        S = cov[0]   #  (B, chunk_size*d, chunk_size*d), use only the first fantasy's covariance (they are the same)
        trace_S2 = torch.sum(S * S, dim=(-2, -1))  # (B,)

        mSm = torch.zeros((mean.shape[0], B), device=mean.device)
        for f in range(mean.shape[0]):
            for b in range(B): 
                m_f_b = mean[f, b]  # (chunk_size*d,)
                Sm =torch.matmul(S[b], m_f_b)   # (chunk_size*d,)
                mSm[f, b] =torch.dot(m_f_b, Sm)

        trace_S2_total += trace_S2
        mSm_total += mSm

        del S, trace_S2, mSm
        torch.cuda.empty_cache()

    mSm_mean = mSm_total.mean(dim=0)   # average over fantasies
    global_var = 2.0 * trace_S2_total + 4.0 * mSm_mean

    return global_var


def _infer_chunk_size_from_device(device: torch.device, d: int, chunk_size: Optional[int]) -> int:
    """
    Empirical chunk size setting. Check the rule of thumb in the theoretical bound in
    G. Lambert, C. Helbert, C. Lauvernet, Gradient-based Active Learning with Gaussian Processes for Global Sensitivity Analysis
    """

    if chunk_size is not None and chunk_size > 0:
        return int(chunk_size)
    if device.type == 'cuda':
        if d <= 5:
            return 500
        if d <= 10:
            return 250
        if d <= 15:
            return 100
        return 50
    else:
        if d <= 5:
            return 2000
        if d <= 10:
            return 1000
        if d <= 15:
            return 500
        return 200


class GlobalGradVarRedKmeans(AcquisitionFunction):
    def __init__(
        self,
        model,
        global_points: Tensor,
        num_fantasies: int = 16,
        chunk_size: Optional[int] =None,
        chunk_kmeans: str= "default",
    ):
        super().__init__(model = model)
        self.global_points =global_points
        self.chunk_kmeans = chunk_kmeans

        device= getattr(self.model, "device", global_points.device)
        d = global_points.shape[-1]
        self.chunk_size = _infer_chunk_size_from_device(device, d, chunk_size)

        self.num_fantasies = num_fantasies
        if d >= 10:
            self.num_fantasies = max(4, num_fantasies // 2)
        self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_fantasies]))

        with torch.no_grad():
            posterior_chunks = self.model.grad_posterior(
                self.global_points,
                chunked =True,
                chunk_size = self.chunk_size,
                chunk_kmeans = self.chunk_kmeans,
            )
            self.curr_var = _global_variance_chunked(posterior_chunks)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        device= X.device
        curr_var =self.curr_var.to(device)
        fantasy_model =self.model.fantasize(Xcond=X, sampler=self.sampler)
        global_points= self.global_points.to(device)
        posterior_fant_chunks =fantasy_model.grad_posterior(
            global_points,
            chunked = True,
            chunk_size =self.chunk_size,
            chunk_kmeans = self.chunk_kmeans,
        )
        lookahead_var =_global_variance_chunked(posterior_fant_chunks).to(device)
        return curr_var - lookahead_var
