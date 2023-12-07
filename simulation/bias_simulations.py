import benign_overfitting as bo
import argparse
import numpy as np
import activation_functions as af

parser = argparse.ArgumentParser(description='Run simulations with varying parameters.')

parser.add_argument('--mu', type=float, nargs='+', default=[1, 100, 200, 500], help='Array of mu values.')
parser.add_argument('--Lambda', type=float, nargs='+', default=[1], help='Array of lambda values.')
parser.add_argument('--n1', type=int, default=30, help='Parameter n1 for symlog points before center point.')
parser.add_argument('--n2', type=int, default=30, help='Parameter n2 for symlog points after center point.')
parser.add_argument('--val', type=int, nargs='+', default=[200], help='Value for Fixed Parameter.')
parser.add_argument('--var', type=str, default='n', help='Which parameter will be fixed? p or n?')
parser.add_argument('--true_p', type=int, default=100, help='True number of predictors.')
parser.add_argument('--tau', type=float, nargs='+', default=[0], help='Array of τ values.')
parser.add_argument('--opt_tau', type=float, nargs='+', default=[], help='Array of τ values.')
parser.add_argument('--tau_grid', type=float, nargs='+', default=[], help='Array of snr values.')
parser.add_argument('--snr', type=int, nargs='+', default=[1, 5, 4], help='Array of snr values.')
parser.add_argument('--sigma', type=float, default=1.0, help='Sigma value.')
parser.add_argument('--test_n', type=int, default=1000, help='Testing set size.')
parser.add_argument('--activation', type=str, default='linear', help='Activation function for nonlinear cases')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')


def simulations_lambda_mu(μ_array, λ_array, n_array, p_array, true_p,
                          τ_array, snr_array, τ_grid, σ, test_n, activation_func,
                          result_arr, progress, seed=None):
    """
    Simulate the test MSE and null risk for different values of λ, μ, n, p, τ.

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
    τ_array : array-like
        Array of values for τ.
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
    max_n = max(n_array)
    max_p = max(p_array)
    ε = bo.compute_ε(σ, max_n+test_n, seed+1) 
    for λ in λ_array:
        for μ in μ_array:
            if activation_func != 'linear':
                X = bo.compute_X_nonlinear(λ, μ, max_n+test_n, max_p, activation_func, seed+2)
            else:
                X = bo.compute_X(λ, μ, max_n+test_n, max_p, seed+2)
            for snr in snr_array:
                β = bo.scale_norm(np.ones(true_p), snr)
                Y = bo.compute_Y(
                    np.ascontiguousarray(X[:, :true_p]), 
                    β, ε)                            
                Y_test = np.ascontiguousarray(Y[max_n:])
                for n in n_array:
                    Y_train = np.ascontiguousarray(Y[:n])
                    for p in p_array:
                        X_train = np.ascontiguousarray(X[:n, :p])
                        X_test = np.ascontiguousarray(X[max_n:, :p])
                        if len(τ_grid) != 0:
                            τ_optimal = bo.find_optimal_tau(X_train, Y_train, τ_grid)
                            print('optimal tau', τ_optimal[0], 'GCV score', τ_optimal[1])
                            τ_array[-1] = τ_optimal[0]
                            print('fitting tau grid', τ_array)
                        for τ in τ_array:
                            params = λ, μ, p, true_p, n, τ, snr
                            β_hat = bo.solve_β_hat(X_train, Y_train, τ)

                            result_arr[idx] = np.array([*params, 
                                                        bo.calculate_MSE(β_hat, X_test, Y_test=Y_test)])
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

    if args.var == 'n':
        n_array = np.array(args.val)
        p_array = np.unique((γ * n_array).astype(int))
    elif args.var == 'p':
        p_array = np.array(args.val)
        n_array = np.unique((γ * p_array).astype(int))
    true_p = args.true_p
    if len(args.tau) < 3:
        print('Fitting Ridgeless Least-squares')
        τ_array = np.array([0])
    else:
        τ_array = np.log(np.arange(args.tau[0], args.tau[1], args.tau[2]))
    if len(args.opt_tau) != 0:
        np.append(τ_array, args.opt_tau)
    if len(args.tau_grid) != 0:
        τ_grid = np.linspace(args.tau_grid[0], args.tau_grid[1], int(args.tau_grid[2]))        
        τ_array = np.append(τ_array, np.nan)
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
    print(f'τ_array: {τ_array}')
    print(f'tau_grid: {τ_grid}')
    print(f'true_p: {true_p}')
    print(f'snr_array: {snr_array}')
    print(f'σ: {σ}')
    print(f'seed: {seed}')

    if len(τ_array) > 0:
        name = f'ridge_bias_{activation_func}_p_{true_p}_'
    else:
        name = f'ridgeless_bias_{activation_func}_p_{true_p}_'

    params = (μ_array, λ_array, n_array, p_array, 
              true_p, τ_array, snr_array, τ_grid, σ, test_n, activation_func)
    bo.run_func_parameters(simulations_lambda_mu, params, 
                        ['λ', 'μ', 'p', 'true_p', 'n', 'tau', 'snr', 'MSE'],
                        seed=seed, name=name)
    
if __name__ == '__main__':
    run_simulations_lambda_mu(parser)