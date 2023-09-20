import numpy as np
from tqdm import tqdm
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import csv
from datetime import datetime
import pandas as pd
import numba
from numba import prange
from numba.typed import List
from numba_progress import ProgressBar
import argparse

@numba.njit(cache=True, fastmath=True, nogil=True)
def solve_β_hat(X, Y):
    """
    Solves the least squares problem using the Moore-Penrose pseudoinverse.

    β_hat = (X^T X)^{+} X^T Y    

    Parameters
    ----------
    X : array-like
        Data matrix for Covariates.
    Y : array-like
        Array for response variable.
    
    Returns
    -------
    β_hat : array-like
        Coefficient vector.
    """

    XTX = X.T @ X
    β_hat = np.linalg.pinv(XTX) @ X.T @ Y
    return β_hat

@numba.njit(cache=True, fastmath=True, nogil=True)
def calculate_MSE(β_hat, β, X_test):
    """
    Calculates the mean squared error of the prediction.

    MSE = (1/n) ||X β_hat - X β||^2

    Parameters
    ----------
    β_hat : array-like
        Coefficient vector.
    β : array-like
        Ground truth coefficient vector.
    X_test : array-like
        Test data matrix for covariates.

    Returns
    -------
    MSE : float
    """
    pred_diff = X_test @ β_hat - X_test @ β
    return np.sum(pred_diff ** 2) / X_test.shape[0]

@numba.njit(cache=True, fastmath=True, nogil=True)
def compute_Y(X, β, ε):
    """
    Computes the response variable Y.

    Y = X β + ε

    Parameters
    ----------
    X : array-like
        Data matrix for Covariates.
    β : array-like
        Coefficient vector.
    ε : array-like
        Noise vector.

    Returns
    -------
    Y : array-like
        Response variable array
    """
    return X @ β + ε

@numba.njit(cache=True, fastmath=True, nogil=True)
def scale_norm(β, snr):
    """
    Scale β to have a given squared l2 norm.

    Parameters
    ----------
    β : array-like
        Vector to be scaled.
    snr : float
        Signal-to-noise ratio.

    Returns
    -------
    β_normalized : array-like
        Normalized β vector with squared l2 norm equal to snr.
    """
    if np.linalg.norm(β) == 0:
        return β
    norm_X = np.linalg.norm(β)**2
    β_normalized = np.sqrt((snr / norm_X)) * β
    return β_normalized

@numba.njit(cache=True, fastmath=True, nogil=True)
def generate_orthonormal_matrix(dim):
    """
    Generate random orthonormal matrix of size dim x dim.

    Parameters
    ----------
    dim : int
        Dimension of the matrix.

    Returns
    -------
    res : array-like
        Orthonormal matrix.
    """
    a = np.ones((dim, dim))
    res, _ = np.linalg.qr(a)
    return np.ascontiguousarray(res)

@numba.njit(cache=True, fastmath=True, nogil=True)
def compute_X(λ, μ, n, p, seed=None):
    """
    Generate X = Γ Z C, where Z is a n x p matrix of iid standard normal 
    random variables;
    
    Γ is a n x n matrix with eigenvalues (μ, 1, ..., 1);
    
    C is a p x p matrix with eigenvalues (λ, 1, ..., 1).

    Parameters
    ----------
    λ : float
        Largest eigenvalue of C.
    μ : float
        Largest eigenvalue of Γ.
    n : int
        Number of observations.
    p : int
        Number of features.
    seed : int
        Seed for the random number generator.
    
    Returns
    -------
    X : array-like
        Data matrix.
    """

    U = generate_orthonormal_matrix(p)
    V = generate_orthonormal_matrix(n)

    Λ = np.diag(np.concatenate((np.array([λ]), np.ones(p-1))))
    C = (U @ Λ) @ U.T
    A = np.diag(np.concatenate((np.array([μ]), np.ones(n-1))))
    Γ = (V @ A) @ V.T
    
    np.random.seed(seed)
    Z = np.random.normal(0, 1, (n, p))
    return Γ @ (Z @ C)

@numba.njit(cache=True, fastmath=True, nogil=True)
def compute_ε(σ, n, seed=None):
    """
    Generate ε = N(0, σ^2 I_n).

    Parameters
    ----------
    σ : float
        Standard deviation of the noise.
    n : int
        Number of observations.
    seed : int
        Seed for the random number generator.

    Returns
    -------
    ε : array-like
        Noise vector.
    """

    np.random.seed(seed)
    return np.random.normal(0, σ, n)

@numba.njit(cache=True, fastmath=True, nogil=True)
def simulate_risks(X, ε, params):
    """
    Fit the LS model and calculate the test MSE and null risk.

    Parameters
    ----------
    X : array-like
        Data matrix.
    ε : array-like
        Noise vector.
    params : tuple
        Tuple of parameters (λ, μ, p, n, snr).\
    
    Returns
    -------
    result : array-like
        Array of parameters and risks.
    """

    λ, μ, p, n, snr = params
    X_p = np.ascontiguousarray(X[:, :p])
    β = scale_norm(np.ones(p), snr)
    Y = compute_Y(X_p, β, ε)
    X_train = np.ascontiguousarray(X_p[:n, :])
    X_test = np.ascontiguousarray(X_p[n:, :])
    Y_train = Y[:n]
    β_hat = solve_β_hat(X_train, Y_train)
    test_MSE = calculate_MSE(β_hat, β, X_test)
    return np.array([λ, μ, p, n, snr, test_MSE], 
                                dtype=np.float64)
    
def efficient_simulation(μ_array, λ_array, n_array, p_array, snr_array, σ, 
                         result_arr, progress, seed=None):
    """
    Simulate the test MSE and null risk for different values of λ, μ, n, p, snr.

    Parameters
    ----------
    μ_array : array-like
        Array of values for μ.
    λ_array : array-like   
        Array of values for λ.
    n_array : array-like
        Array of values for n.
    p_array : array-like
        Array of values for p.
    snr_array : array-like
        Array of values for snr.
    σ : float
        Standard deviation of the noise.
    result_arr : array-like
        Array to store the results.
    progress : ProgressBar
        Progress bar.
    seed : int
        Seed for the random number generator.
    
    Returns
    -------
    result_arr : array-like
        Array of parameters and risks.
    """

    if seed is None:
        raise ValueError('seed is None')
    idx = 0
    n = max(n_array)
    max_p = max(p_array)
    test_n = 10000
    ε = compute_ε(σ, n+test_n, seed+1) 
    for λ in λ_array:
        for μ in μ_array:
            X = compute_X(λ, μ, n+test_n, max_p, seed+2)
            for snr in snr_array:
                for p in p_array:
                    params = λ, μ, p, n, snr
                    result_arr[idx] = simulate_risks(X, ε, params)
                    idx += 1
                    progress.update(1)
    return result_arr

def generate_symlog_points(n1, n2, L, U, a):
    """
    Generate a list of points in a symmetric logarithmic scale.
    
    The points are generated in the interval [L, U] with a break at a.

    Parameters
    ----------
    n1 : int
        Number of points in the interval [L, a).
    n2 : int
        Number of points in the interval (a, U].
    L : float
        Lower bound of the interval.
    U : float
        Upper bound of the interval.
    a : float
        Break point.
    
    Returns
    -------
    symlog_points : array-like
        Array of points in a symmetric logarithmic scale.
    """

    log_part_lower = np.logspace(np.log10(L), np.log10(a-0.001), n1, endpoint=False)
    log_part_upper = np.logspace(np.log10(a+0.001), np.log10(U), n2, endpoint=True)
    symlog_points = np.concatenate([log_part_lower, log_part_upper])
    
    return symlog_points

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run simulations with varying parameters.')
    
    parser.add_argument('--mu', type=int, nargs='+', default=[1, 100, 200, 500], help='Array of mu values.')
    parser.add_argument('--Lambda', type=int, nargs='+', default=[1], help='Array of lambda values.')
    parser.add_argument('--n1', type=int, default=30, help='Parameter n1 for symlog points before center point.')
    parser.add_argument('--n2', type=int, default=30, help='Parameter n2 for symlog points after center point.')
    parser.add_argument('--n_array', type=int, nargs='+', default=[200], help='Array of n values.')
    parser.add_argument('--snr', type=int, nargs='+', default=[1, 5, 4], help='Array of snr values.')
    parser.add_argument('--sigma', type=float, default=1.0, help='Sigma value.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    
    args = parser.parse_args()
    
    μ_array = np.array(args.mu)
    λ_array = np.array(args.Lambda)
    n1, n2 = args.n1, args.n2
    γ = generate_symlog_points(n1, n2, 0.1, 10, 1)
    n_array = np.array(args.n_array)
    p_array = np.unique((γ * n_array).astype(int))
    snr_array = np.linspace(args.snr[0], args.snr[1], args.snr[2])
    σ = args.sigma
    seed = args.seed

    print('Running Simulations')
    print(f'μ_array: {μ_array}')
    print(f'λ_array: {λ_array}')
    print(f'n_array: {n_array}')
    print(f'p_array: {p_array}')
    print(f'snr_array: {snr_array}')
    print(f'σ: {σ}')
    print(f'seed: {seed}')

    start_time = time.time()
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    print("date and time =", dt_string)
    filename = f'results/Python/results_[{dt_string}-{seed}].csv'
    total_combinations = len(μ_array) * len(λ_array) * len(n_array) * len(p_array) * len(snr_array)
    result_arr = np.zeros((total_combinations, 6), dtype=np.float64)
    with ProgressBar(total=total_combinations) as progress:
        result_arr = efficient_simulation(μ_array, λ_array, n_array, p_array, snr_array, σ, result_arr, progress, seed=seed)
    df = pd.DataFrame(result_arr, columns=['λ', 'μ', 'p', 'n', 'snr', 'MSE'])
    df.to_csv(filename, index=False)
    print(time.time()-start_time)
    print('Finished Runing Simulations')
