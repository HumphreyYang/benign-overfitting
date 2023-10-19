import benign_overfitting as bo
import argparse
import numpy as np
import activation_functions as af

parser = argparse.ArgumentParser(description='Run simulations with varying parameters.')

parser.add_argument('--mu', type=float, nargs='+', default=[1, 100, 200, 500], help='Array of mu values.')
parser.add_argument('--Lambda', type=float, nargs='+', default=[1], help='Array of lambda values.')
parser.add_argument('--n1', type=int, default=30, help='Parameter n1 for symlog points before center point.')
parser.add_argument('--n2', type=int, default=30, help='Parameter n2 for symlog points after center point.')
parser.add_argument('--n_array', type=int, nargs='+', default=[200], help='Array of n values.')
parser.add_argument('--snr', type=int, nargs='+', default=[1, 5, 4], help='Array of snr values.')
parser.add_argument('--sigma', type=float, default=1.0, help='Sigma value.')
parser.add_argument('--test_n', type=int, default=1000, help='Testing set size.')
parser.add_argument('--activation', type=str, default='linear', help='Activation function for nonlinear cases')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')


def simulations_lambda_mu(μ_array, λ_array, n_array, p_array, 
                          snr_array, σ, test_n, activation_func,
                          result_arr, progress, seed=None):
    """
    Simulate the test MSE and null risk for different values of λ, μ, n, p, snr.

    Parameters
    ----------
    μ_array : array-like
        Array of values for μ.
    λ_array : array-like   
        Array of values for λ.
    n_array : array-like
        Array of values for n.
    p_array : array-like
        Array of values for p.
    snr_array : array-like
        Array of values for snr.
    σ : float
        Standard deviation of the noise.
    result_arr : array-like
        Array to store the results.
    progress : ProgressBar
        Progress bar.
    seed : int
        Seed for the random number generator.
    
    Returns
    -------
    result_arr : array-like
        Array of parameters and risks.
    """

    if seed is None:
        raise ValueError('seed is None')
    idx = 0
    n = max(n_array)
    max_p = max(p_array)
    ε = bo.compute_ε(σ, n+test_n, seed+1) 
    for λ in λ_array:
        for μ in μ_array:
            if activation_func != 'linear':
                X = bo.compute_X_nonlinear(λ, μ, n+test_n, max_p, activation_func, seed+2)
            else:
                X = bo.compute_X(λ, μ, n+test_n, max_p, seed+2)
            for snr in snr_array:
                for p in p_array:
                    params = λ, μ, p, n, snr
                    result_arr[idx] = np.array([*params, bo.simulate_risks(X, ε, p, n, snr)])
                    idx += 1
                    progress.update(1)
    return result_arr


def run_simulations_lambda_mu(parser):
    """
    Main function for running simulations with varying μ and λ.

    Parameters
    ----------

    parser : argparse.ArgumentParser
        Parser for command line arguments.
    
    Returns
    -------
    None

    Results written in are written into CSV files with name defined by the name
    provided in run_func_parameters function.
    """
    args = parser.parse_args()
    
    μ_array = np.array(args.mu)
    λ_array = np.array(args.Lambda)
    n1, n2 = args.n1, args.n2
    γ = bo.generate_symlog_points(n1, n2, 0.1, 10, 1)
    n_array = np.array(args.n_array)
    p_array = np.unique((γ * n_array).astype(int))
    snr_array = np.linspace(args.snr[0], args.snr[1], args.snr[2])
    σ = args.sigma
    activation_func = args.activation
    test_n = args.test_n
    seed = args.seed

    print('Running Simulations')
    print(f'μ_array: {μ_array}')
    print(f'λ_array: {λ_array}')
    print(f'n_array: {n_array}')
    print(f'p_array: {p_array}')
    print(f'snr_array: {snr_array}')
    print(f'σ: {σ}')
    print(f'seed: {seed}')

    params = μ_array, λ_array, n_array, p_array, snr_array, σ, test_n, activation_func
    bo.run_func_parameters(simulations_lambda_mu, params, 
                        ['λ', 'μ', 'p', 'n', 'snr', 'MSE'],
                        seed=seed, name=f'lambda_mu_{activation_func}_')
    
if __name__ == '__main__':
    run_simulations_lambda_mu(parser)