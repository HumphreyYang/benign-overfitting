import numpy as np
from scipy.stats import ortho_group
from numba import jit


def solve_β_hat(X, Y):    
    β_hat = np.linalg.pinv(X.T @ X) @ X.T @ Y
    return β_hat

def calculate_MSE(β_hat, X, Y):
    MSE = np.sum((Y - X @ β_hat)**2) / len(Y)
    return MSE

def compute_Y(X, β, σ, seed=0):
    np.random.seed(seed)
    ε = np.random.normal(0, σ, len(X))
    return X.T @ β + ε

def compute_X(λ, μ, p, n, seed=0):
    C = compute_C(λ, p, seed)
    Γ = compute_Γ(μ, n, seed)
    
    np.random.seed(seed)
    Z = np.random.normal(0, 1, (p, n))

    print('C = ', C.shape)
    print('Z = ', Z.shape)
    print('Γ = ', Γ.shape)

    return C @ Z @ Γ

def compute_C(λ, p, seed):
    np.random.seed(seed)
    U = ortho_group.rvs(dim=(p))
    U = U.reshape((p, p))
    Λ = compute_Λ(λ, p)
    return U @ Λ @ U.T

def compute_Γ(μ, n, seed):
    np.random.seed(seed)
    V = ortho_group.rvs(dim=(n))
    V = V.reshape((n, n))
    A = compute_A(μ, n)
    print('V = ', V.shape)
    print('A = ', A.shape)
    return V @ A @ V.T

def compute_Λ(λ, p):
    λ_array = np.repeat(1, p)
    λ_array[0] = λ
    return np.diag(λ_array)

def compute_A(μ, n):
    μ_array = np.repeat(1, n)
    μ_array[0] = μ
    return np.diag(μ_array)


def simulate_test_MSE(λ, μ, p, n, seed=0):
    X = compute_X(λ, μ, p, n, seed=seed)
    print('X = ', X.shape)
    print('cov(X) = ', np.cov(X, rowvar=False))

    # Train-test Split
    X_train = X[:int(n*0.7), :]
    X_test = X[int(n*0.7):, :]
    print('n = ', n)
    print('training size = ', int(n*0.7))
    print('X_train = ', X_train.shape)
    print('X_test = ', X_test.shape)

    # Compute Y
    β = np.repeat(1, p)
    σ = 1
    Y = compute_Y(X, β, σ, seed=seed)
    Y_train = Y[:int(n*0.7)]
    Y_test = Y[int(n*0.7):]
    print('Y_train = ', Y_train.shape)
    print('Y_test = ', Y_test.shape)

    # Compute β_hat
    β_hat = solve_β_hat(X_train, Y_train)
    print('β_hat = ', β_hat.shape)

    # Compute MSE
    MSE_train = calculate_MSE(β_hat, X_train, Y_train)
    print('Training MSE = ', MSE_train)

    # Compute MSE_test
    MSE_test = calculate_MSE(β_hat, X_test, Y_test)
    print('Testing MSE= ', MSE_test)

    return MSE_test

def run_simulations(μ_array, λ_array, n_array, p_array):
    MSE_matrix = np.zeros((len(μ_array), len(λ_array), len(n_array), len(p_array)))

    for a, μ in enumerate(μ_array):
        for b, λ in enumerate(λ_array):
            for c, n in enumerate(n_array):
                for d, p in enumerate(p_array):
                    print('γ =', p / n)
                    MSE_matrix[a, b, c, d] = simulate_test_MSE(λ, μ, p, n, seed=0)

    return MSE_matrix

def vectorized_run_simulations(μ_array, λ_array, n_array, p_array):

    # Create grids for all parameters
    μ_grid, λ_grid, n_grid, p_grid = np.meshgrid(μ_array, λ_array, n_array, p_array, indexing='ij')
    
    # Calculate γ values
    γ_values = p_grid / n_grid
    print('γ_values:', γ_values)
    
    # Vectorize the function
    vec_simulate_test_MSE = np.vectorize(simulate_test_MSE)
    
    # Apply function to entire grid
    MSE_matrix = vec_simulate_test_MSE(λ_grid, μ_grid, p_grid, n_grid, seed=0)
    
    return MSE_matrix


if __name__ == "__main__":

    μ_array = np.linspace(0.5, 100, 100)
    λ_array = np.linspace(0.5, 100, 100)
    n_array = np.arange(10, 110, 10)
    p_array = np.arange(10, 110, 10)

    MSE_matrix = run_simulations(μ_array, λ_array, n_array, p_array)
    np.save('mse_matrix.npy', MSE_matrix)





    

