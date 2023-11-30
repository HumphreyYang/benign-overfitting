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
from collections.abc import Iterable
from statsmodels.stats.correlation_tools import cov_nearest
import activation_functions as af

def solve_β_hat(X, Y, τ=0):
    """
    Solves the least squares problem using the Moore-Penrose pseudoinverse.

    β_hat = (X^T X + nτI)^{+} X^T Y    
    
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
    if τ == 0:
        β_hat = np.linalg.pinv(XTX) @ X.T @ Y
    else:
        n = X.shape[0]
        I = np.eye(X.shape[1])
        β_hat = np.linalg.inv(XTX + n*τ*I) @ X.T @ Y
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
def calculate_MSE_Y(β_hat, Y_test, X_test):
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
    pred_diff = X_test @ β_hat - Y_test
    return np.sum(pred_diff ** 2) / X_test.shape[0]

def is_pos_semidef(X, ϵ=1e-5):
    return np.all(np.linalg.eigvals(X) >= -ϵ)

def compute_compound_cov(p, ρ, σ):
    Σ_compound = np.diag(np.ones(p))
    mask_off_dia = ~np.eye(Σ_compound.shape[0],dtype=bool)
    Σ_compound[mask_off_dia] = ρ
    Σ_compound = σ**2 * Σ_compound
    Σ_compound = cov_nearest(Σ_compound)
    assert is_pos_semidef(Σ_compound)

    return Σ_compound

def compute_random_compound_cov(p, ρ_mag, σ):
    Σ_compound = np.diag(np.ones(p))
    mask_off_dia = ~np.eye(Σ_compound.shape[0],dtype=bool)
    Σ_compound[mask_off_dia] = np.random.uniform(-ρ_mag, ρ_mag, p**2 - p)
    Σ_compound = σ**2 * Σ_compound
    Σ_compound = cov_nearest(Σ_compound)
    assert is_pos_semidef(Σ_compound)

    return Σ_compound


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
    # a = np.random.uniform(10, 20, (dim, dim))
    a = np.ones((dim, dim))
    res, _ = np.linalg.qr(a)
    return np.ascontiguousarray(res)

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
    C = ((U @ Λ) @ U.T).real
    A = np.diag(np.concatenate((np.array([μ]), np.ones(n-1))))
    Γ = ((V @ A) @ V.T).real

    # assert is_pos_semidef(Γ)
    # assert is_pos_semidef(C)
    
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
    return np.random.normal(0.0, σ, n)

def simulate_risks(X, ε, p, n, max_n, τ, snr):
    """
    Fit the LS model and calculate the test MSE.

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

    β = scale_norm(np.ones(p), snr)
    X_train = np.ascontiguousarray(X[:n, :p])
    X_test = np.ascontiguousarray(X[max_n:, :p])
    Y_train = compute_Y(X_train, β, ε)
    β_hat = solve_β_hat(X_train, Y_train, τ)
    test_MSE = calculate_MSE(β_hat, β, X_test)
    return test_MSE

@numba.njit(cache=True, fastmath=True, nogil=True)
def check_pos_simidef(X):
    return np.all(np.linalg.eigvals(X) >= 0)

def compute_X_nonlinear(λ, μ, n, p, ϕ, seed=None):
    if ϕ == 'abs':
        ϕ = af.phi_abs
    elif ϕ == 'quad':
        ϕ = af.phi_quad
    else:
        raise ValueError('Invalid activation function')
        
    U = generate_orthonormal_matrix(p)
    V = generate_orthonormal_matrix(n)

    Λ = np.diag(np.concatenate((np.array([λ]), np.ones(p-1))))
    C = ((U @ Λ) @ U.T).real
    A = np.diag(np.concatenate((np.array([μ]), np.ones(n-1))))
    Γ = ((V @ A) @ V.T).real

    np.random.seed(seed)

    X = np.zeros((n, p))

    for i in range(n):
        z = np.random.normal(0, 1, p)
        x_i = ϕ(C @ z)
        X[i, :] = x_i
        
    return Γ @ X

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

def average_iterations(filename):
    print(filename)
    df = pd.read_csv(filename+'.csv')
    df = df.groupby(['λ', 'μ', 'p', 'n', 'snr']).mean().reset_index()
    df.to_csv(filename+'_average.csv', index=False)

def run_func_parameters(func, params, columns, seed=None, name=''):
    start_time = time.time()
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    print("date and time =", dt_string)
    print(columns)
    filename = f'results/Python/{name}results_[{dt_string}-{seed}]'
    print(filename)
    total_com = 1
    for param in params:
        total_com *= len(param) if hasattr(param,  '__len__') and type(param) is not str else 1
    print((total_com, len(columns)))
    result_arr = np.zeros((total_com, len(columns)), dtype=np.float64)
    with ProgressBar(total=total_com) as progress:
        print(params)
        result_arr = func(*params, result_arr, progress, seed=seed)
    df = pd.DataFrame(result_arr, columns=columns)
    df.to_csv(filename+'.csv', index=False)
    print(time.time()-start_time)
    print('Finished Runing Simulations')

    print(filename)

    # average_iterations(filename)