import benign_overfitting as bo
import numpy as np


def compute_X_compound(ρ, σ, n, p, random, seed=None):

    if random:
        Σ = bo.compute_random_compound_cov(p, ρ, σ)
    else:
        Σ = bo.compute_compound_cov(p, ρ, σ)

    Γ = np.diag(np.ones(n))
    
    np.random.seed(seed)
    Z = np.random.normal(0, 1, (n, p))
    return Γ @ (Z @ Σ)


def simulations_compound(ρ_array, n_array, p_array, snr_array, σ, 
                         result_arr, progress, seed=None):
    
    if seed is None:
        raise ValueError('seed is None')
    idx = 0
    n = max(n_array)
    max_p = max(p_array)
    test_n = 10000
    ε = bo.compute_ε(σ, n+test_n, seed+1) 
    for ρ in ρ_array:
        X = compute_X_compound(ρ, 1, n+test_n, max_p, True, seed+2)
        for snr in snr_array:
            for p in p_array:
                params = ρ, p, n, snr
                result_arr[idx] = np.array([*params, bo.simulate_risks(X, ε, p, n, snr)])
                idx += 1
                progress.update(1)

    return result_arr


if __name__ == '__main__':
    # ρ_array = np.array([-0.3, -0.5, -0.7])
    ρ_array = np.array([0.3, 0.5, 1])
    n1, n2 = 30, 30
    γ = bo.generate_symlog_points(n1, n2, 0.1, 100, 1)
    n_array = np.array([200])
    p_array = np.unique((γ * n_array).astype(int))
    snr_array = np.linspace(1, 5, 4)
    σ = 1.0

    params = ρ_array, n_array, p_array, snr_array, σ
    bo.run_func_parameters(simulations_compound, params, 
                           ['ρ', 'p', 'n', 'snr', 'MSE'], 
                           seed=1, name='compound_random')