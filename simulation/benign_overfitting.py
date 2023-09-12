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


def solve_β_hat(X, Y):
    XTX = X.T @ X
    β_hat = np.linalg.pinv(XTX) @ X.T @ Y
    return β_hat


def calculate_MSE(β_hat, X, Y):
    pred_diff = Y - X @ β_hat
    return np.sum(pred_diff ** 2) / Y.shape[0]


def scale_norm(X, out_norm):
    if np.linalg.norm(X) == 0:
        return X
    norm_X = np.linalg.norm(X)
    X_normalized = (out_norm / norm_X) * X
    return X_normalized


def compute_X(λ, μ, n, p, seed=None):
    """
    Compute X based on the formula:
    X = Γ Z C

    where Γ = V A V^T, A = diag(μ, 1, ..., 1) with shape (n x n)
          C = U Λ U^T, Λ = diag(λ, 1, ..., 1) with shape (p x p)
    
    and Z ~ N(0, I_{n x p})
    """
    if seed is not None:
        U = generate_orthonormal_matrix(p, seed=seed+1)
        V = generate_orthonormal_matrix(n, seed=seed+2)
    else:
        U = generate_orthonormal_matrix(p)
        V = generate_orthonormal_matrix(n)

    A = np.diag(np.concatenate((np.array([μ]), np.ones(n-1))))
    Γ = (V @ A) @ V.T
    Λ = np.diag(np.concatenate((np.array([λ]), np.ones(p-1))))
    C = (U @ Λ) @ U.T
    
    if seed is not None:
        np.random.seed(seed)
        
    Z = np.random.normal(0, 1, (n, p))
    return Γ @ (Z @ C)


def generate_orthonormal_matrix(dim, seed=None):
    """
    Generate a random orthonormal matrix of shape (dim x dim)
    """

    if seed is not None:
        np.random.seed(seed)
    a = np.random.randn(dim, dim)
    res, _ = np.linalg.qr(a)
    return np.ascontiguousarray(res)


def compute_Y(X, β, ε):
    return X @ β + ε

def simulate_test_MSE(X, ε, p, n, snr):
    X_p = np.ascontiguousarray(X[:, :p])
    train_size = int(0.7 * n)
    X_train, X_test = np.split(X_p, [train_size])
    X_train = np.ascontiguousarray(X_train)
    X_test = np.ascontiguousarray(X_test)
    β = scale_norm(np.ones(p), snr)
    Y = compute_Y(X_p, β, ε)
    Y_train, Y_test = np.split(Y, [train_size])

    β_hat = solve_β_hat(X_train, Y_train)

    # print('*' * 80)
    # print(f'summary of parameters:p={p}, n={n}')
    # print(f'summary of shapes: X shape={X_p.shape}, Y shape={Y.shape}, \
    #         X_train shape={X_train.shape}, X_test shape={X_test.shape}, β_hat shape={β_hat.shape}, \
    #         norm_β_hat={np.linalg.norm(β_hat)}, norm_β={np.linalg.norm(β)}')
    # print(f'time taken = {time.time() - start_time} seconds')
    # print('*' * 80)

    return calculate_MSE(β_hat, X_test, Y_test)


def compute_ε(shape, σ, seed=None):
    if seed is not None:
        np.random.seed(seed)
    ε = np.random.normal(0, σ, shape)
    return ε
 
def simulate_test_MSE_for_grid(X, ε, params):
    _, _, p, n, snr, _ = params
    simulation_result = simulate_test_MSE(X, ε, p, n, snr)
    return simulation_result

def paralleled_numba(X, ε, param_list, func):
    result_arr = np.zeros((len(param_list), 6))
    
    for idx in tqdm(range(len(param_list))):
        result_arr[idx, :-1] = np.array(param_list[idx][:-1], dtype=np.float64)
        result_arr[idx, -1] = func(X, ε, param_list[idx])

    return result_arr

def run_simulations(μ_array, λ_array, n_array, p_array, snr_array, σ, seed=None):

    ε = compute_ε(np.max(n_array), σ, seed=seed)
    simulation_count = len(μ_array) * len(λ_array) * len(n_array) * len(p_array) * len(snr_array)
    result_arr = np.zeros((simulation_count, 6))
    for λ in λ_array:
        for μ in μ_array:
            X = compute_X(λ, μ, n=np.max(n_array), p=np.max(p_array), seed=seed)

            np.random.shuffle(p_array)

            param_list = [(λ, μ, p, n, snr, seed)
                            for n in n_array
                            for p in p_array
                            for snr in snr_array]
            
            for idx in tqdm(range(len(param_list))):
                params = param_list[idx]
                result_arr[idx, :-1] = np.array(params[:-1], dtype=np.float64)
                result_arr[idx, -1] = simulate_test_MSE_for_grid(X, ε, params)
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
    filename = f'results/Python/results_[{dt_string}-{seed}].csv'
    num_iterations = len(μ_array) * len(λ_array) * len(n_array) * len(p_array) * len(snr_array)
    result_arr = run_simulations(μ_array, λ_array, n_array, p_array, snr_array, σ, seed=seed)
    df = pd.DataFrame(result_arr, columns=['λ', 'μ', 'p', 'n', 'snr', 'MSE'])
    df.to_csv(filename, index=False)
    print(time.time()-start_time)
    print('Finished Runing Simulations')
