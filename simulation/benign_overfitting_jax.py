import jax
import numpy as np
import jax.numpy as jnp
from jax import random, jit, vmap
import time
from datetime import datetime
import pandas as pd
import csv
from tqdm import tqdm
from functools import partial

jax.config.update('jax_platform_name', 'cpu')

@jit
def solve_β_hat(X, Y):
    XTX = X.T @ X
    β_hat = jnp.linalg.pinv(XTX) @ X.T @ Y
    return β_hat

@jit
def calculate_MSE(β_hat, X, Y):
    pred_diff = Y - X @ β_hat
    return jnp.sum(pred_diff ** 2) / Y.shape[0]

@jit
def compute_Y(X, β, σ, key):
    ε = σ * random.normal(key, (X.shape[0],))
    return X @ β + ε

@partial(jax.jit, static_argnums=(2, 3))
def compute_X(λ, μ, p, n, U, V, key):
    Λ = jnp.diag(jnp.concatenate([jnp.array([λ]), jnp.ones(p-1)]))
    C = (U @ Λ) @ U.T
    A = jnp.diag(jnp.concatenate([jnp.array([μ]), jnp.ones(n-1)]))
    Γ = V @ A @ V.T
    Z = random.normal(key, (n, p))
    return Γ @ (Z @ C)

@jit
def scale_norm(X, out_norm):
    norm_X = jnp.linalg.norm(X)
    X_normalized = (out_norm / norm_X) * X
    return X_normalized

@partial(jax.jit, static_argnums=(0, ))
def generate_orthonormal_matrix(dim, key):
    a = random.normal(key, (dim, dim))
    res, _ = jnp.linalg.qr(a)
    return res

@partial(jax.jit, static_argnums=(2, 3))
def simulate_test_MSE(λ, μ, p, n, snr, key):
    subkeys = random.split(key, 3)
    U = generate_orthonormal_matrix(p, subkeys[0])
    V = generate_orthonormal_matrix(n, subkeys[1])

    X = compute_X(λ, μ, p, n, U, V, subkeys[2])
    
    train_size = int(0.7 * n)
    X_train, X_test = jnp.split(X, [train_size])
    β = scale_norm(jnp.ones(p), snr)
    σ = 1.0
    
    Y = compute_Y(X, β, σ, key)
    Y_train, Y_test = jnp.split(Y, [train_size])
    β_hat = solve_β_hat(X_train, Y_train)
    
    return calculate_MSE(β_hat, X_test, Y_test)

def vectorized_run_simulations(μ_array, λ_array, n_array, p_array):
    μ_grid, λ_grid, n_grid, p_grid = jnp.meshgrid(μ_array, λ_array, n_array, p_array, indexing='ij')
    vec_simulate_test_MSE = jax.jit(vmap(simulate_test_MSE, in_axes=(None, None, None, None, None, 0)))
    keys = random.split(random.PRNGKey(0), μ_grid.size)
    return vec_simulate_test_MSE(λ_grid.ravel(), μ_grid.ravel(), p_grid.ravel(), n_grid.ravel(), jnp.ones(λ_grid.size), keys)

def run_and_save_simulations(μ_array, λ_array, n_array, p_array, snr, seed):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    key = random.PRNGKey(seed)
    param_list = [(λ, μ, p, n, snr, key) 
                for μ in μ_array
                for λ in λ_array
                for n in n_array
                for p in p_array]
    with open(f'results/results_[{dt_string}].csv', 'w+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['μ', 'λ', 'n', 'p', 'snr' 'MSE'])
        for param in tqdm(param_list):
            mse = simulate_test_MSE(*param)
            csvwriter.writerow([*param[:-1], mse])

if __name__ == "__main__":
    μ_array = np.linspace(1, 20, 40)
    λ_array = np.linspace(1, 20, 40)
    γ = np.linspace(0.05, 5.05, 500)
    n_array = np.array([100])
    p_array = np.unique((γ * n_array).astype(int))
    snr = 1.0
    seed = 1046

    start_time = time.time()
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    print("date and time =", dt_string)

    result = run_and_save_simulations(μ_array, λ_array, n_array, p_array, snr, seed)
    
    # # Saving to a Pandas DataFrame and then to CSV
    # df = pd.DataFrame(result.ravel(), columns=['MSE'])
    # df.to_csv(f'results/results_[{dt_string}].csv', index=False)

    print(time.time()-start_time)
    print('Finished Running Simulations')
