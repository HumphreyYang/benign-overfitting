import numpy as np
from scipy.stats import ortho_group
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import csv
from datetime import datetime
import numba
from scipy.stats import ortho_group

@numba.njit(cache=True, fastmath=True)
def solve_β_hat(X, Y):
    XTX = X.T @ X
    β_hat = np.linalg.pinv(XTX) @ X.T @ Y
    return β_hat

@numba.njit(cache=True, fastmath=True, parallel=True)
def calculate_MSE(β_hat, X, Y):
    pred_diff = Y - X @ β_hat
    return np.sum(pred_diff ** 2) / len(Y)

@numba.njit(cache=True, fastmath=True)
def compute_Y(X, β, σ, seed=None):
    if seed is not None:
        np.random.seed(seed)
    ε = np.random.normal(0, σ, len(X))
    return X @ β + ε

@numba.njit(cache=True, fastmath=True)
def compute_X(λ, μ, p, n, U, V, seed=None):
    Λ = np.diag(np.concatenate((np.array([λ]), np.ones(p-1))))
    C = (U @ Λ) @ U.T
    A = np.diag(np.concatenate((np.array([μ]), np.ones(n-1))))
    Γ = (V @ A) @ V.T
    
    if seed is not None:
        np.random.seed(seed)
        
    Z = np.random.normal(0, 1, (n, p))
    return Γ @ (Z @ C)

@numba.njit(cache=True, fastmath=True, parallel=True)
def scale_norm(X, out_norm):
    if np.linalg.norm(X) == 0:
        return X
    norm_X = np.linalg.norm(X)
    X_normalized = (out_norm / norm_X) * X
    return X_normalized

@numba.njit(cache=True, fastmath=True)
def generate_orthonormal_matrix(dim, seed=None):
    if seed is not None:
        np.random.seed(seed)
    a = np.random.uniform(0.5, 1, size=(dim, dim))
    res, _ = np.linalg.qr(a)
    return np.ascontiguousarray(res)

@numba.njit(cache=True, fastmath=True)
def simulate_test_MSE(λ, μ, p, n, snr, seed=None):
    if seed is not None:
        U = generate_orthonormal_matrix(p, seed=seed+1)
        V = generate_orthonormal_matrix(n, seed=seed+2)
    else:
        U = generate_orthonormal_matrix(p)
        V = generate_orthonormal_matrix(n)
    
    X = compute_X(λ, μ, p, n, U, V, seed)
    
    train_size = int(0.7 * n)
    X_train, X_test = np.split(X, [train_size])
    X_train = np.ascontiguousarray(X_train)
    X_test = np.ascontiguousarray(X_test)

    β = scale_norm(np.ones(p), snr)
    σ = 1.0
    
    Y = compute_Y(X, β, σ)
    Y_train, Y_test = np.split(Y, [train_size])

    β_hat = solve_β_hat(X_train, Y_train)

    # print('*' * 80)
    # print(f'summary of parameters: λ={λ}, μ={μ}, p={p}, n={n}')
    # print(f'summary of shapes: X shape={X.shape}, Y shape={Y.shape}, \
    #         X_train shape={X_train.shape}, X_test shape={X_test.shape}, β_hat shape={β_hat.shape}, \
    #         norm_β_hat={np.linalg.norm(β_hat)}, norm_β={np.linalg.norm(β)}')
    # print(f'time taken = {time.time() - start_time} seconds')
    # print('*' * 80)

    return calculate_MSE(β_hat, X_test, Y_test)

def vectorized_run_simulations(μ_array, λ_array, n_array, p_array):
    μ_grid, λ_grid, n_grid, p_grid = np.meshgrid(μ_array, λ_array, n_array, p_array, indexing='ij')
    vec_simulate_test_MSE = np.vectorize(simulate_test_MSE)
    return vec_simulate_test_MSE(λ_grid, μ_grid, p_grid, n_grid)

def simulate_test_MSE_for_grid(params):
    λ, μ, p, n, snr, seed = params
    simulation_result = simulate_test_MSE(λ, μ, p, n, snr, seed=seed)
    return simulation_result

def paralleled_compute(param_list, simulate_test_MSE_for_grid, csvwriter):
    if len(param_list) == 0:
        return
    
    batch_size = 50  # Adjust based on available memory and the expected size of data
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
            
def parallel_run_simulations_to_csv(μ_array, λ_array, n_array, p_array, snr, seed=None, parallel=True, filename='results.csv'):
    with open(filename, 'w+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['λ', 'μ', 'p', 'n', 'snr', 'MSE'])  # Write the header row

        param_list = [(λ, μ, p, n, snr, seed) 
                        for μ in μ_array
                        for λ in λ_array
                        for n in n_array
                        for p in p_array]

        if parallel:
            paralleled_compute(param_list, simulate_test_MSE_for_grid, csvwriter)
        else:
            for params in tqdm(param_list):
                csvwriter.writerow([*params[:-1], simulate_test_MSE_for_grid(params)])
    
    return None

if __name__ == "__main__":
    μ_array = np.linspace(1, 20, 50)
    λ_array = np.linspace(1, 20, 50)
    γ = np.linspace(0.05, 5.05, 500)
    n_array = np.array([100])
    p_array = np.unique((γ * n_array).astype(int))
    snr = 5.0
    seed = 1525 # current time

    start_time = time.time()
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    print("date and time =", dt_string)

    parallel_run_simulations_to_csv(μ_array, λ_array, n_array, p_array, snr, seed=seed, 
                                    parallel=False, filename=f'results/results_[{dt_string}-{seed}].csv')
    print(start_time - time.time())
    print('Finished Runing Simulations')