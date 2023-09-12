import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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

@numba.njit(cache=True, fastmath=True, parallel=True, nogil=True)
def calculate_MSE(β_hat, X, Y):
    pred_diff = Y - X @ β_hat
    return np.sum(pred_diff ** 2) / len(Y)

@numba.njit(cache=True, fastmath=True, nogil=True)
def compute_Y(X, β, ε):
    return X @ β + ε

@numba.njit(cache=True, fastmath=True, parallel=True, nogil=True)
def scale_norm(X, out_norm):
    if np.linalg.norm(X) == 0:
        return X
    norm_X = np.linalg.norm(X)
    X_normalized = (out_norm / norm_X) * X
    return X_normalized

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

@numba.njit(cache=True, fastmath=True, nogil=True)
def compute_MSE(X, ε, p, snr, train_size):
    X_p = np.ascontiguousarray(X[:, :p])
    X_train, X_test = X_p[:train_size], X_p[train_size:]
    β = scale_norm(np.ones(p), snr)
    Y = compute_Y(X_p, β, ε)
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    β_hat = solve_β_hat(X_train, Y_train)
    test_MSE = calculate_MSE(β_hat, X_test, Y_test)
    return test_MSE

def efficient_simulation(μ_array, λ_array, n_array, p_array, snr_array, σ, 
                         result_arr, progress, seed=None):
    if seed is None:
        raise ValueError('seed is None')
    idx = 0
    max_n = max(n_array)
    max_p = max(p_array)
    for λ in λ_array:
        for μ in μ_array:
            X = compute_X(λ, μ, max_n, max_p, seed)
            ε = compute_ε(σ, max_n, seed)
            for n in n_array:
                train_size = int(0.7 * n)
                for p in p_array:
                    for snr in snr_array:
                        test_MSE = compute_MSE(X, ε, p, snr, train_size)
                        result_arr[idx] = np.array([λ, μ, p, n, snr, test_MSE], dtype=np.float64)
                        idx += 1
                        progress.update(1)
    return result_arr

if __name__ == "__main__":
    μ_array = np.linspace(1, 100, 8)
    λ_array = np.array([1])
    γ = [*np.linspace(0.1, 0.9, 25), *np.linspace(1.1, 10, 25)]
    print(γ)
    n_array = np.array([200])
    p_array = np.unique((γ * n_array).astype(int))
    snr_array = np.linspace(1, 5, 4)
    σ = 1.0
    print(snr_array)
    seed = 2011

    start_time = time.time()
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    print("date and time =", dt_string)
    filename = f'results/results_[{dt_string}-{seed}].csv'
    total_combinations = len(μ_array) * len(λ_array) * len(n_array) * len(p_array) * len(snr_array)
    result_arr = np.zeros((total_combinations, 6), dtype=np.float64)
    with ProgressBar(total=total_combinations) as progress:
        result_arr = efficient_simulation(μ_array, λ_array, n_array, p_array, snr_array, σ, result_arr, progress, seed=seed)
    df = pd.DataFrame(result_arr, columns=['λ', 'μ', 'p', 'n', 'snr', 'MSE'])
    df.to_csv(filename, index=False)
    print(time.time()-start_time)
    print('Finished Runing Simulations')
