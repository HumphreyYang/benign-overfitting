import numpy as np
from scipy.stats import ortho_group
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import csv
from datetime import datetime
import numba

@numba.njit(cache=True, fastmath=True)
def check_orthonormal(A):
    n, m = A.shape

    # Check if columns are unit vectors
    col_norms = np.linalg.norm(A, ord=2)
    if not np.allclose(col_norms, 1, atol=1e-8, rtol=1e-8, equal_nan=False):
       return False
    
    # Check orthogonality
    ortho_check = np.dot(A.T, A)
    if not np.allclose(ortho_check, np.eye(n), atol=1e-8, rtol=1e-8, equal_nan=False):
       return False
    
    return True

@numba.njit(cache=True, fastmath=True)
def solve_β_hat(X, Y):
    XTX = X.T @ X
    β_hat = np.linalg.pinv(XTX) @ X.T @ Y
    return β_hat

@numba.njit(cache=True, fastmath=True)
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
def compute_X(λ, μ, p, n, seed=None):
    if seed is not None:
        np.random.seed(seed)
        C = compute_C(λ, p, seed+1)
        Γ = compute_Γ(μ, n, seed+2)
    else:
        C = compute_C(λ, p)
        Γ = compute_Γ(μ, n)
    Z = np.random.normal(0, 1, (p, n))
    return C @ Z @ Γ

@numba.njit(cache=True, fastmath=True)
def compute_C(λ, p, seed=None):
    if seed is not None:
        np.random.seed(seed)
    a = np.random.random(size=(p, p))
    U, _ = np.linalg.qr(a)
    assert check_orthonormal(U), 'not orthonormal'
    Λ = np.diag(np.concatenate((np.array([λ]), np.ones(p-1))))
    return U @ Λ @ U.T

@numba.njit(cache=True, fastmath=True)
def compute_Γ(μ, n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    a = np.random.random(size=(n, n))
    V, _ = np.linalg.qr(a)
    assert check_orthonormal(V), 'not orthonormal'
    A = np.diag(np.concatenate((np.array([μ]), np.ones(n-1))))
    return V @ A @ V.T

@numba.njit(cache=True, fastmath=True)
def scale_norm(X, out_norm):
    norm_X = np.linalg.norm(X)
    X_normalized = (out_norm / norm_X) * X
    return X_normalized

@numba.njit(cache=True, fastmath=True)
def simulate_test_MSE(λ, μ, p, n, snr, seed=None):
    # start_time = time.time()
    X = compute_X(λ, μ, p, n, seed).T
    train_size = int(0.7 * n)
    X_train, X_test = np.split(X, [train_size])
    
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
    λ, μ, p, n, snr = params
    simulation_result = simulate_test_MSE(λ, μ, p, n, snr, seed=0)
    return simulation_result

# def parallel_run_simulations(μ_array, λ_array, n_array, p_array, snr):
#     MSE_matrix = np.zeros((len(μ_array), len(λ_array), len(n_array), len(p_array)))

#     param_list = [(a, b, c, d, λ, μ, p, n, snr) 
#                   for a, μ in enumerate(μ_array)
#                   for b, λ in enumerate(λ_array)
#                   for c, n in enumerate(n_array)
#                   for d, p in enumerate(p_array)]

#     with ProcessPoolExecutor(max_workers=7) as executor:
#         results = executor.map(simulate_test_MSE_for_grid, param_list)

#     for (a, b, c, d, λ, μ, p, n, snr), result in zip(param_list, results):
#         try:
#             MSE_matrix[a, b, c, d] = result
#         except Exception as e:
#             print(f"An exception occurred: {e}")

#     return MSE_matrix

def parallel_run_simulations_to_csv(μ_array, λ_array, n_array, p_array, snr, parallel=True, filename='results.csv'):
    with open(filename, 'w+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['λ', 'μ', 'p', 'n', 'snr', 'MSE'])  # Write the header row
        
        param_list = [(λ, μ, p, n, snr) 
                        for μ in μ_array
                        for λ in λ_array
                        for n in n_array
                        for p in p_array]
        if parallel:
            future_list = []
            with ProcessPoolExecutor() as executor:
                print('submitting jobs to executor')
                for params in tqdm(param_list):
                    print('submitting params: ', params)
                    future = executor.submit(simulate_test_MSE_for_grid, params)
                    future_list.append([params, future])

                for params_future in future_list:
                    params = params_future[0]
                    future = params_future[1]
                    try:
                        csvwriter.writerow([*params, future.result()])
                    except Exception as e:
                        print(e)
        else:
            for params in tqdm(param_list):
                csvwriter.writerow([*params, simulate_test_MSE_for_grid(params)])
    
    return None

if __name__ == "__main__":
    μ_array = np.linspace(1, 100, 100)
    λ_array = np.linspace(1, 100, 100)
    γ = np.linspace(0.5, 1.5, 200)
    n_array = np.array([100])
    p_array = (γ * n_array).astype(int)
    snr = 5

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    print("date and time =", dt_string)

    # MSE_matrix = parallel_run_simulations(μ_array, λ_array, n_array, p_array, snr)
    # np.save(f'mse_matrix_{μ_param}_{λ_param}_{γ_param}_{n_param}_{snr_param}.npy', MSE_matrix)

    parallel_run_simulations_to_csv(μ_array, λ_array, n_array, p_array, snr, parallel=False, filename=f'results_[{dt_string}].csv')
    print('Finished Runing Simulations')
