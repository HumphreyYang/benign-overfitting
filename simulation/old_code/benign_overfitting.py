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
def compute_Y(X, β, σ, seed=None):
    if seed is not None:
        np.random.seed(seed)
    ε = np.random.normal(0, σ, len(X))
    return X @ β + ε

@numba.njit(cache=True, fastmath=True, nogil=True)
def compute_X(λ, μ, p, n, U, V, seed=None):
    Λ = np.diag(np.concatenate((np.array([λ]), np.ones(p-1))))
    C = (U @ Λ) @ U.T
    A = np.diag(np.concatenate((np.array([μ]), np.ones(n-1))))
    Γ = (V @ A) @ V.T
    
    if seed is not None:
        np.random.seed(seed)
        
    Z = np.random.normal(0, 1, (n, p))
    return Γ @ (Z @ C)

@numba.njit(cache=True, fastmath=True, parallel=True, nogil=True)
def scale_norm(β, snr):
    if np.linalg.norm(β) == 0:
        return β
    norm_X = np.linalg.norm(β)**2
    β_normalized = np.sqrt((snr / norm_X)) * β
    return β_normalized

@numba.njit(cache=True, fastmath=True, nogil=True)
def generate_orthonormal_matrix(dim, seed=None):
    if seed is not None:
        np.random.seed(seed)
    a = np.random.randn(dim, dim)
    res, _ = np.linalg.qr(a)
    return np.ascontiguousarray(res)

@numba.njit(cache=True, fastmath=True, nogil=True)
def simulate_test_MSE(λ, μ, p, n, snr, test_n, seed=None):
    if seed is not None:
        U = generate_orthonormal_matrix(p, seed=seed+1)
        V = generate_orthonormal_matrix(n*10, seed=seed+2)
    else:
        U = generate_orthonormal_matrix(p)
        V = generate_orthonormal_matrix(n*10)
    
    X = compute_X(λ, μ, p, n*10, U, V, seed)
    
    X_train, X_test = np.split(X, [n])
    X_train = np.ascontiguousarray(X_train)
    X_test = np.ascontiguousarray(X_test)

    β = scale_norm(np.ones(p), snr)
    σ = 1.0
    
    Y = compute_Y(X, β, σ)
    null_risk =  np.sum(Y - np.mean(Y) ** 2) / len(Y)
    Y_train, Y_test = np.split(Y, [n])

    β_hat = solve_β_hat(X_train, Y_train)

    # print('*' * 80)
    # print(f'summary of parameters: λ={λ}, μ={μ}, p={p}, n={n}')
    # print(f'summary of shapes: X shape={X.shape}, Y shape={Y.shape}, \
    #         X_train shape={X_train.shape}, X_test shape={X_test.shape}, β_hat shape={β_hat.shape}, \
    #         norm_β_hat={np.linalg.norm(β_hat)}, norm_β={np.linalg.norm(β)}')
    # print(f'time taken = {time.time() - start_time} seconds')
    # print('*' * 80)

    return calculate_MSE(β_hat, X_test, Y_test), null_risk

def vectorized_run_simulations(μ_array, λ_array, n_array, p_array):
    μ_grid, λ_grid, n_grid, p_grid = np.meshgrid(μ_array, λ_array, n_array, p_array, indexing='ij')
    vec_simulate_test_MSE = np.vectorize(simulate_test_MSE)
    return vec_simulate_test_MSE(λ_grid, μ_grid, p_grid, n_grid)

@numba.njit(cache=True, fastmath=True, nogil=True)
def simulate_test_MSE_for_grid(params):
    λ, μ, p, n, snr, test_n, seed = params
    simulation_result = simulate_test_MSE(λ, μ, p, n, snr, test_n, seed=seed)
    return simulation_result

def paralleled_compute(param_list, simulate_test_MSE_for_grid, csvwriter):
    if len(param_list) == 0:
        return
    
    batch_size = 50
    results_buffer = []
    
    with ThreadPoolExecutor() as executor:
        print('submitting jobs to executor')
        
        future_to_params = {executor.submit(simulate_test_MSE_for_grid, params): params for params in param_list}
        
        print('start collecting results')
        
        for future in tqdm(as_completed(future_to_params)):
            params = future_to_params[future]
            try:
                result = future.result()
                
                results_buffer.append([*params[:-1], result])
                
                if len(results_buffer) >= batch_size:
                    csvwriter.writerows(results_buffer)
                    results_buffer.clear()
                    
            except Exception as e:
                print(f"Exception occurred with params {params}: {e}")
        
        if results_buffer:
            csvwriter.writerows(results_buffer)

def paralleled_numba(typed_param_list, func, progress):
    result_arr = np.zeros((len(typed_param_list), 7))
    
    for idx in prange(len(typed_param_list)):
        result_arr[idx, :-2] = np.array(typed_param_list[idx][:-2], dtype=np.float64)
        result_arr[idx, -2:] = func(typed_param_list[idx])
        progress.update(1)

    return result_arr

def parallel_run_simulations(μ_array, λ_array, n_array, p_array, snr_array, test_n, seed=None, native_parallel=True, filename='results.csv'):
    param_list = [(λ, μ, p, n, snr, test_n, seed) 
                    for μ in μ_array
                    for λ in λ_array
                    for n in n_array
                    for p in p_array
                    for snr in snr_array]

    if native_parallel:
        with open(filename, 'w+', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['λ', 'μ', 'p', 'n', 'snr', 'MSE', 'null_risk']) 
            paralleled_compute(param_list, simulate_test_MSE_for_grid, csvwriter)
    else:
        typed_param_list = List()
        for p in param_list:
            typed_param_list.append(p)
        with ProgressBar(total=len(param_list)) as progress:
            result_arr = paralleled_numba(typed_param_list, simulate_test_MSE_for_grid, progress)
        df = pd.DataFrame(result_arr, columns=['λ', 'μ', 'p', 'n', 'snr', 'MSE', 'null_risk'])
        df.to_csv(filename, index=False)
    return None

if __name__ == "__main__":
    μ_array = np.array([1])
    λ_array = np.array([1])
    γ = [*np.linspace(0.1, 0.9, 25), *np.linspace(1.1, 10, 25)]
    print(γ)
    n_array = np.array([200])
    p_array = np.unique((γ * n_array).astype(int))
    print(p_array)
    snr_array = np.array([5])
    σ = 1.0
    print(snr_array)
    seed = 2355
    test_n = 10000

    start_time = time.time()
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    print("date and time =", dt_string)

    parallel_run_simulations(μ_array, λ_array, n_array, p_array, snr_array, test_n, seed=seed, 
                                    native_parallel=False, filename=f'results/Python/results_[{dt_string}-{seed}].csv')
    print(time.time()-start_time)
    print('Finished Runing Simulations')
