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

@numba.njit(cache=True, fastmath=True, nogil=True)
def solve_β_hat(X, Y):
    XTX = X.T @ X
    β_hat = np.linalg.pinv(XTX) @ X.T @ Y
    return β_hat

def calculate_MSE(β_hat, X, Y):
    pred_diff = Y - (X @ β_hat)
    return np.sum(pred_diff ** 2) / X.shape[0]

@numba.njit(cache=True, fastmath=True, nogil=True)
def compute_Y(X, β, ε):
    return X @ β + ε

@numba.njit(cache=True, fastmath=True, parallel=True, nogil=True)
def scale_norm(β, snr):
    if np.linalg.norm(β) == 0:
        return β
    norm_X = np.linalg.norm(β)
    β_normalized = (snr / norm_X) * β
    return β_normalized

@numba.njit(cache=True, fastmath=True, nogil=True)
def generate_orthonormal_matrix(dim, seed=None):
    np.random.seed(seed)
    a = np.random.randn(dim, dim)
    res, _ = np.linalg.qr(a)
    return np.ascontiguousarray(res)

@numba.njit(cache=True, fastmath=True, nogil=True)
def compute_X(λ, μ, n, p, seed=None):

    U = generate_orthonormal_matrix(p, seed=seed)
    V = generate_orthonormal_matrix(n, seed=seed)

    Λ = np.diag(np.concatenate((np.array([λ]), np.ones(p-1))))
    C = (U @ Λ) @ U.T
    A = np.diag(np.concatenate((np.array([μ]), np.ones(n-1))))
    Γ = (V @ A) @ V.T
    
    np.random.seed(seed)
        
    Z = np.random.normal(0, 1, (n, p))
    return Γ @ (Z @ C)

@numba.njit(cache=True, fastmath=True, nogil=True)
def compute_ε(σ, n, seed=None):
    np.random.seed(seed)
    return np.random.normal(0, σ, n)

def efficient_simulation(μ_array, λ_array, n_array, p_array, snr_array, σ, 
                         result_arr, progress, seed=None):
    if seed is None:
        raise ValueError('seed is None')
    idx = 0
    n = max(n_array)
    max_p = max(p_array)
    test_n = 1000
    ε = compute_ε(σ, n+test_n, seed+1) 
    for λ in λ_array:
        for μ in μ_array:
            X = compute_X(λ, μ, n+test_n, max_p, seed)
            for snr in snr_array:
                for p in p_array:
                    X_p = np.ascontiguousarray(X[:, :p])
                    β = scale_norm(np.ones(p), snr)
                    Y = compute_Y(X_p, β, ε)
                    X_train = X_p[:n,:]
                    X_test = X_p[n:, :]
                    Y_train = Y[:n]
                    Y_test = Y[n:]
                    null_risk = calculate_MSE(np.zeros(max_p), X, Y)
                    β_hat = solve_β_hat(X_train, Y_train)
                    test_MSE = calculate_MSE(β_hat, X_test, Y_test)
                    result_arr[idx] = np.array([λ, μ, p, n, snr, test_MSE, null_risk], 
                                                dtype=np.float64)
                    idx += 1
                    progress.update(1)
    return result_arr


# Corrected function to generate symlog spaced points focusing on non-negative side
def generate_symlog_points(n1, n2, L, U, a):
    # Generate points for the logarithmic part from [L, a)
    log_part_lower = np.logspace(np.log10(L), np.log10(a-0.001), n1, endpoint=False)
    
    # Generate points for the logarithmic part from (a, U]
    log_part_upper = np.logspace(np.log10(a+0.001), np.log10(U), n2, endpoint=True)
    
    # Concatenate all three parts
    symlog_points = np.concatenate([log_part_lower, log_part_upper])
    
    return symlog_points

if __name__ == "__main__":
    μ_array = np.array([1])
    λ_array = np.array([1])
    n1, n2 = 30, 30
    γ = generate_symlog_points(n1, n2, 0.1, 10, 1)

    print(γ)
    
    n_array = np.array([200])
    p_array = np.unique((γ * n_array).astype(int))
    print(p_array)
    snr_array = np.linspace(1, 5, 4)
    σ = 1.0
    print(snr_array)
    seed = 2355

    start_time = time.time()
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    print("date and time =", dt_string)
    filename = f'results/Python/results_[{dt_string}-{seed}].csv'
    total_combinations = len(μ_array) * len(λ_array) * len(n_array) * len(p_array) * len(snr_array)
    result_arr = np.zeros((total_combinations, 7), dtype=np.float64)
    with ProgressBar(total=total_combinations) as progress:
        result_arr = efficient_simulation(μ_array, λ_array, n_array, p_array, snr_array, σ, result_arr, progress, seed=seed)
    df = pd.DataFrame(result_arr, columns=['λ', 'μ', 'p', 'n', 'snr', 'MSE', 'null_risk'])
    df.to_csv(filename, index=False)
    print(time.time()-start_time)
    print('Finished Runing Simulations')
