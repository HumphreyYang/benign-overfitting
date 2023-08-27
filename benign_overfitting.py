import numpy as np
from scipy.stats import ortho_group
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def solve_β_hat(X, Y):
    XTX = X.T @ X
    β_hat = np.linalg.pinv(XTX) @ X.T @ Y
    return β_hat

def calculate_MSE(β_hat, X, Y):
    pred_diff = Y - X @ β_hat
    return np.sum(pred_diff ** 2) / len(Y)

def compute_Y(X, β, σ, seed=0):
    if seed is not None:
        np.random.seed(seed)
    ε = np.random.normal(0, σ, len(X))
    return X @ β + ε

def compute_X(λ, μ, p, n, seed=0):
    C = compute_C(λ, p, seed)
    Γ = compute_Γ(μ, n, seed)
    if seed is not None:
        np.random.seed(seed)
    Z = np.random.normal(0, 1, (p, n))
    return C @ Z @ Γ

def compute_C(λ, p, seed=0):
    if seed is not None:
        np.random.seed(seed)
    U = ortho_group.rvs(dim=(p))
    Λ = np.diag(np.concatenate(([λ], np.ones(p-1))))
    return U @ Λ @ U.T

def compute_Γ(μ, n, seed=0):
    if seed is not None:
        np.random.seed(seed)
    V = ortho_group.rvs(dim=(n))
    A = np.diag(np.concatenate(([μ], np.ones(n-1))))
    return V @ A @ V.T

def simulate_test_MSE(λ, μ, p, n, seed=0):
    start_time = time.time()
    X = compute_X(λ, μ, p, n, seed=seed).T
    train_size = int(0.7 * n)
    X_train, X_test = np.split(X, [train_size])
    
    β = np.ones(p)
    σ = 1.0
    Y = compute_Y(X, β, σ, seed=seed)
    Y_train, Y_test = np.split(Y, [train_size])

    β_hat = solve_β_hat(X_train, Y_train)

    print('*' * 80)
    print(f'summary of parameters: λ={λ}, μ={μ}, p={p}, n={n}', )
    print(f'summary of shapes: X shape={X.shape}, Y shape={Y.shape}, X_train shape={X_train.shape}, Y_train shape={X_test.shape}, β_hat shape={β_hat.shape}')
    print(f'time taken = {time.time() - start_time:.2f} seconds')
    print('*' * 80)
    
    return calculate_MSE(β_hat, X_test, Y_test)



def vectorized_run_simulations(μ_array, λ_array, n_array, p_array):
    μ_grid, λ_grid, n_grid, p_grid = np.meshgrid(μ_array, λ_array, n_array, p_array, indexing='ij')
    vec_simulate_test_MSE = np.vectorize(simulate_test_MSE)
    return vec_simulate_test_MSE(λ_grid, μ_grid, p_grid, n_grid)

def simulate_test_MSE_for_grid(params):
    a, b, c, d, λ, μ, p, n = params
    return simulate_test_MSE(λ, μ, p, n, seed=0)

def parallel_run_simulations(μ_array, λ_array, n_array, p_array):
    MSE_matrix = np.zeros((len(μ_array), len(λ_array), len(n_array), len(p_array)))

    param_list = [(a, b, c, d, λ, μ, p, n) 
                  for a, μ in enumerate(μ_array)
                  for b, λ in enumerate(λ_array)
                  for c, n in enumerate(n_array)
                  for d, p in enumerate(p_array)]

    with ProcessPoolExecutor() as executor:
        results = executor.map(simulate_test_MSE_for_grid, param_list)

    for (a, b, c, d), result in zip(param_list, results):
        try:
            MSE_matrix[a, b, c, d] = result
        except Exception as e:
            print(f"An exception occurred: {e}")

    return MSE_matrix

if __name__ == "__main__":
    np.random.seed(0)  # Set seed once
    μ_array = np.linspace(1, 100, 5)
    λ_array = np.linspace(1, 100, 5)
    γ = np.linspace(0.5, 100, 5)
    n_array = np.array([100])
    p_array = (γ * n_array).astype(int)

    MSE_matrix = parallel_run_simulations(μ_array, λ_array, n_array, p_array)
    np.save('benign-overfitting/mse_matrix.npy', MSE_matrix)
