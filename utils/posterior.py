import math
import torch
from gpytorch.distributions import MultivariateNormal


def batched_posterior_derivative_joint_fantasize(model, Xs, return_full: bool =True):
    r"""
    Computes the joint gradient posterior for a (fantasized) GP model.

    Args:
        model: A fantasized GP model with the following assumptions:
               - model.train_inputs[0] has shape (n_fant, B, n_train, d) OR can be
                 unbatched (n_train, d) or candidate-batched (B, n_train, d). In these cases,
                 the inputs are normalized to (n_fant, B, n_train, d).
               - model.train_targets has shape (n_fant, B, n_train)
               - model.likelihood.noise has shape (B, n_train)
               - model.mean_module.constant is a scalar.
               - The kernel is ARD Matern 5/2 with parameters:
                     lengthscale: tensor of shape (d,)
                     outputscale: scalar
        Xs: Test (global) input points of shape (N, d)
        return_full: If False, return only the flattened gradient mean without computing the covariance.

    Returns:
        If return_full is True (default): MultivariateNormal with
          - mean of shape (n_fant, B, N*d)
          - covariance_matrix of shape (n_fant, B, N*d, N*d)
        Otherwise: Tensor of shape (n_fant, B, N*d) containing the flattened gradient mean.

    Based on Gpytorch Matern52Grad Kernel implementation.
    """
    device =Xs.device
    dtype = Xs.dtype
    N, d = Xs.shape

    #Pre processing to ensure train inputs/oupust are (n_fant, B, n_train, d)
    train_in = model.train_inputs[0]
    if train_in.ndim == 2:
        train_in= train_in.unsqueeze(0).unsqueeze(0)  # (1, 1, n_train, d)
    elif train_in.ndim == 3:
        if train_in.shape[0] > 1:
            train_in= train_in.unsqueeze(1)   # (n_fant, 1, n_train, d)
        else:
            train_in = train_in.unsqueeze(0) # (1, B, n_train, d)
    elif train_in.ndim == 4:
        pass
    else:
        raise ValueError("Unsupported number of dimensions for model.train_inputs[0].")

    y_train = model.train_targets
    if y_train.ndim == 1:
        y_train= y_train.unsqueeze(0).unsqueeze(0)   # (1, 1, n_train)
    elif y_train.ndim == 2:
        if y_train.shape[0] > 1:
            y_train = y_train.unsqueeze(1)   # (n_fant, 1, n_train)
        else:
            y_train= y_train.unsqueeze(0)  # (1, B, n_train)
    elif y_train.ndim == 3:
        pass
    else:
        raise ValueError("Unsupported number of dimensions for model.train_targets.")

    # Pre proccess for noise shape (B, n_train)
    noise = model.likelihood.noise
    if noise.ndim == 0:
        noise= noise.expand(train_in.shape[1], train_in.shape[2])
    elif noise.ndim == 1:
        noise = noise.unsqueeze(0).expand(train_in.shape[1], train_in.shape[2])
    elif noise.ndim == 2:
        pass
    else:
        raise ValueError("Unsupported number of dimensions for model.likelihood.noise.")

    n_fant, B, n_train, d_train = train_in.shape
    assert d == d_train, "Dimension mismatch between test and training inputs."

    m_val = model.mean_module.constant.detach()  # assumed constant mean
    lengthscales = model.covar_module.base_kernel.lengthscale.squeeze()  # (d,)
    outputscale = model.covar_module.outputscale.item()   # scalar

    K_train = model.covar_module(train_in, train_in).to_dense() #(n_fant, B, n_train, n_train)
    eye_n = torch.eye(n_train, dtype=dtype, device=device)
    noise = noise.clamp_min(1e-6)
    K_train = K_train + noise.unsqueeze(0).unsqueeze(-1) * eye_n
    jitter = 1e-6
    L= torch.linalg.cholesky(K_train + jitter * eye_n) # (n_fant, B, n_train, n_train)

    diff_train =y_train - m_val   # (n_fant, B, n_train)
    M_diff = torch.cholesky_solve(diff_train.unsqueeze(-1), L).squeeze(-1) # (n_fant, B, n_train), solve L * x = diff_train for each (fantasy, candidate) pair.


    # Compute Gradient kernel between Xs and X_train 
    Xs_exp = Xs.unsqueeze(0).unsqueeze(0).unsqueeze(3)   # (1, 1, N, 1, d)
    X_train_exp = train_in.unsqueeze(2) # (n_fant, B, 1, n_train, d)
    diff = Xs_exp - X_train_exp # shape (n_fant, B, N, n_train, d)

    scaled_diff = diff / lengthscales.view(1, 1, 1, 1, d)
    sqrt5 = torch.sqrt(torch.tensor(5.0, dtype=dtype, device=device))
    r= torch.norm(scaled_diff, dim=-1)  # shape (n_fant, B, N, n_train)

    ### Derivative factor for Matern52 ###
    factor = -5 * outputscale / 3 * (1 + sqrt5 * r) * torch.exp(-sqrt5 * r)    # (n_fant, B, N, n_train)
    # grad_K: derivative of kernel, shape (n_fant, B, N, n_train, d)
    grad_K = factor.unsqueeze(-1) * (diff / (lengthscales**2).view(1, 1, 1, 1, d))

    #  Posterior Gradient Mean #
    ## Based on the remark that : grad_mean[f, b, i, d] = sum_{j} grad_K[f, b, i, j, d] * M_diff[f, b, j]
    grad_mean= torch.einsum('fbikd,fbk->fbid', grad_K, M_diff)  # (n_fant, B, N, d)
    grad_mean_flat= grad_mean.reshape(n_fant, B, N * d) #(n_fant, B, N*d)

    if not return_full:
        return grad_mean_flat

    ###Compute Hessian kernel ###
    diff_tt =Xs.unsqueeze(1) - Xs.unsqueeze(0) #(N, N, d)
    scaled_diff_tt = diff_tt / lengthscales.view(1, 1, d)  # (N, N, d)
    r_tt= torch.norm(scaled_diff_tt, dim=-1) # (N, N)
    h_val_tt =(1 + sqrt5 * r_tt) * torch.exp(-sqrt5 * r_tt) # (N, N)
    outer_tt = diff_tt.unsqueeze(-1) * (diff_tt / (lengthscales**2).view(1, 1, d)).unsqueeze(-2) # (N, N, d, d)
    A = (5 * outputscale / 3) / (lengthscales**2) # (d,)
    eye_d =torch.eye(d, dtype=dtype, device=device)
    exp_factor = 5 * torch.exp(-sqrt5 * r_tt)
    # H_prior: (N, N, d, d)
    H_prior = - A.view(1, 1, d, 1) * (exp_factor.unsqueeze(-1).unsqueeze(-1) * outer_tt -
                                       h_val_tt.unsqueeze(-1).unsqueeze(-1) * eye_d)
    H_prior_batched= H_prior.unsqueeze(0).unsqueeze(0).expand(n_fant, B, N, N, d, d)   # (n_fant, B, N, N, d, d)

    ## Compute the cross-term of Cov ##
    # Solve L * X = grad_K for each test point.
    L_exp = L.unsqueeze(2).expand(n_fant, B, N, n_train, n_train)  # (n_fant, B, N, n_train, n_train)
    B_sol = torch.cholesky_solve(grad_K, L_exp)  # (n_fant, B, N, n_train, d)

    # For each (f, b) and test points i and j, cross_term[f, b, i, j, a, b] = sum_{k} grad_K[f, b, i, k, a] * B_sol[f, b, j, k, b]
    cross_term= torch.einsum('fbikd,fbjke->fbijde', grad_K, B_sol)  # (n_fant, B, N, N, d, d)

    #  posterior grad hessian matrix
    # cov_grad = H_prior_batched - cross_term, shape (n_fant, B, N, N, d, d)
    cov_grad = H_prior_batched - cross_term
    cov_grad = cov_grad.permute(0, 1, 2, 4, 3, 5).reshape(n_fant, B, N * d, N * d)
    cov_grad= 0.5 * (cov_grad + cov_grad.transpose(-1, -2))
    cov_grad =cov_grad + 1e-6 * torch.eye(N*d, device=device, dtype=dtype).expand(n_fant, B, N*d, N*d)

    return MultivariateNormal(grad_mean_flat, cov_grad)
    # the cov_grad computation is based on the followin :
    ### Rearrange the block structure so that each test point contributes d entries: 
    # Permute from (n_fant, B, i, j, a, b) to (n_fant, B, i, a, j, b) and reshape to (n_fant, B, N*d, N*d)

from k_means_constrained import KMeansConstrained
def _resolve_chunk_size_points(d: int, device_type: str, requested: int= None) -> int:
    """Keep chunk_size*d bounded to avoid blowing up the covariance blocks."""
    base = 500 if device_type == "cuda" else 2000
    cap = max(1, base // max(1, d))
    if requested is None:
        return cap
    requested = int(max(1, requested))
    return min(requested, cap)

def batched_posterior_derivative_joint_fantasize_chunked_kmeans(
    model,
    Xs,
    chunk_size: int = 100,
    selection_alg: str = "kmeans-equal",
    verbose: bool =False,
):
    """
    Adaptation of batched_posterior_derivative_joint_fantasize to chunk approximation
    More details in :
    G. Lambert, C. Helbert, C. Lauvernet : Gradient-based Active Learning with Gaussian Processes for Global Sensitivity Analysis
    """

    device= Xs.device
    dtype =Xs.dtype
    N, d = Xs.shape

    chunk_size= _resolve_chunk_size_points(d, device.type, chunk_size)
    n_chunks = max(1, (N + chunk_size - 1) // chunk_size)
    labels= None
    Xs_np= Xs.detach().cpu().numpy()

    if selection_alg in ("kmeans-equal", "balanced"):
        if verbose:
            print("Performing balanced k-means clustering...")
        kmeans = KMeansConstrained(
            n_clusters =n_chunks,
            size_min = N // n_chunks,
            size_max = math.ceil(N / n_chunks),
            init = "k-means++",
            max_iter = 2000,
            tol = 1e-5,
            n_jobs =-1,
        )
        kmeans.fit(Xs_np)
        labels = kmeans.labels_
    elif selection_alg in ("kmeans", "default"):
        from sklearn.cluster import KMeans

        if verbose:
            print("Performing classical k-means clustering...")
        kmeans = KMeans(
            n_clusters = n_chunks,
            init = "k-means++",
            max_iter = 2000,
        )
        labels =kmeans.fit_predict(Xs_np)
    else:
        raise ValueError(
            "Unsupported selection algorithm. "
            "Use 'default' or 'balanced'."
        )

    train_in =model.train_inputs[0]
    if train_in.ndim == 2:
        train_in= train_in.unsqueeze(0).unsqueeze(0)  # (1, 1, n_train, d)
    elif train_in.ndim == 3:
        if train_in.shape[0] > 1:
            train_in = train_in.unsqueeze(1)  # (n_fant, 1, n_train, d)
        else:
            train_in= train_in.unsqueeze(0) # (1, B, n_train, d)
    elif train_in.ndim == 4:
        pass
    else:
        raise ValueError("Unsupported dimensions for model.train_inputs[0].")

    y_train = model.train_targets
    if y_train.ndim == 1:
        y_train = y_train.unsqueeze(0).unsqueeze(0)    # (1, 1, n_train)
    elif y_train.ndim == 2:
        if y_train.shape[0] > 1:
            y_train =y_train.unsqueeze(1)  # (n_fant, 1, n_train)
        else:
            y_train = y_train.unsqueeze(0)    # (1, B, n_train)
    elif y_train.ndim == 3:
        pass
    else:
        raise ValueError("Unsupported dimensions for model.train_targets.")

    noise = model.likelihood.noise
    if noise.ndim == 0:
        noise = noise.expand(train_in.shape[1], train_in.shape[2])
    elif noise.ndim == 1:
        noise = noise.unsqueeze(0).expand(train_in.shape[1], train_in.shape[2])
    elif noise.ndim == 2:
        pass
    else:
        raise ValueError("Unsupported dimensions for model.likelihood.noise.")

    n_fant, B, n_train, d_train= train_in.shape
    assert d == d_train, "Dimension mismatch between test and training inputs."

    m_val= model.mean_module.constant.detach()
    lengthscales= model.covar_module.base_kernel.lengthscale.squeeze()
    outputscale= model.covar_module.outputscale.item()

    K_train = model.covar_module(train_in, train_in).to_dense()
    eye_n = torch.eye(n_train, dtype=dtype, device=device)
    K_train = K_train + noise.unsqueeze(0).unsqueeze(-1) * eye_n
    L = torch.linalg.cholesky(K_train)

    diff_train= y_train - m_val
    M_diff= torch.cholesky_solve(diff_train.unsqueeze(-1), L).squeeze(-1)

    sqrt5 =torch.sqrt(torch.tensor(5.0, dtype=dtype, device=device))

    A= (5 * outputscale / 3) / (lengthscales**2)
    eye_d = torch.eye(d, dtype=dtype, device=device)

    chunk_results = []

    for i in range(n_chunks):
        if selection_alg in ("kmeans-equal", "balanced", "kmeans", "default"):
            cluster_mask= torch.as_tensor(labels == i, device=device, dtype=torch.bool)
            chunk_N =int(cluster_mask.sum().item())
            if chunk_N == 0:
                continue
            Xs_chunk= Xs[cluster_mask]
        else:
            raise ValueError("Unsupported selection algorithm. Select a valid option.")


        Xs_exp = Xs_chunk.unsqueeze(0).unsqueeze(0).unsqueeze(3)
        X_train_exp = train_in.unsqueeze(2)
        diff =Xs_exp - X_train_exp

        scaled_diff = diff / lengthscales.view(1, 1, 1, 1, d)
        r= torch.norm(scaled_diff, dim=-1)

        factor = -5 * outputscale / 3 * (1 + sqrt5 * r) * torch.exp(-sqrt5 * r)
        grad_K =factor.unsqueeze(-1) * (diff / (lengthscales**2).view(1, 1, 1, 1, d))

        grad_mean = torch.einsum('fbikd,fbk->fbid', grad_K, M_diff)
        grad_mean_flat = grad_mean.reshape(n_fant, B, chunk_N * d)

        diff_tt = Xs_chunk.unsqueeze(1) - Xs_chunk.unsqueeze(0)
        scaled_diff_tt = diff_tt / lengthscales.view(1, 1, d)
        r_tt =torch.norm(scaled_diff_tt, dim=-1)
        h_val_tt = (1 + sqrt5 * r_tt) * torch.exp(-sqrt5 * r_tt)
        outer_tt = diff_tt.unsqueeze(-1) * (diff_tt / (lengthscales**2).view(1, 1, d)).unsqueeze(-2)

        exp_factor = 5 * torch.exp(-sqrt5 * r_tt)
        H_prior = - A.view(1, 1, d, 1) * (exp_factor.unsqueeze(-1).unsqueeze(-1) * outer_tt -
                                           h_val_tt.unsqueeze(-1).unsqueeze(-1) * eye_d)
        H_prior_batched =H_prior.unsqueeze(0).unsqueeze(0).expand(n_fant, B, chunk_N, chunk_N, d, d)

        L_exp= L.unsqueeze(2).expand(n_fant, B, chunk_N, n_train, n_train)
        B_sol = torch.cholesky_solve(grad_K, L_exp)
        cross_term = torch.einsum('fbikd,fbjke->fbijde', grad_K, B_sol)

        cov_grad = H_prior_batched - cross_term
        cov_grad = cov_grad.permute(0, 1, 2, 4, 3, 5).reshape(n_fant, B, chunk_N * d, chunk_N * d)

        chunk_results.append((grad_mean_flat, cov_grad))

        if device.type == 'cuda':
            del Xs_exp, X_train_exp, diff, scaled_diff, r, factor, grad_K
            del H_prior_batched, cross_term, cov_grad, grad_mean, grad_mean_flat
            torch.cuda.empty_cache()
        else:
            del Xs_exp, X_train_exp, diff, scaled_diff, r, factor, grad_K
            del H_prior_batched, cross_term, cov_grad, grad_mean, grad_mean_flat
    return chunk_results
