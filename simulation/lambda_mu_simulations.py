import benign_overfitting as bo
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Run simulations with varying parameters.')

parser.add_argument('--mu', type=int, nargs='+', default=[1, 100, 200, 500], help='Array of mu values.')
parser.add_argument('--Lambda', type=int, nargs='+', default=[1], help='Array of lambda values.')
parser.add_argument('--n1', type=int, default=30, help='Parameter n1 for symlog points before center point.')
parser.add_argument('--n2', type=int, default=30, help='Parameter n2 for symlog points after center point.')
parser.add_argument('--n_array', type=int, nargs='+', default=[200], help='Array of n values.')
parser.add_argument('--snr', type=int, nargs='+', default=[1, 5, 4], help='Array of snr values.')
parser.add_argument('--sigma', type=float, default=1.0, help='Sigma value.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')


def run_simulations_lambda_mu(parser):
    args = parser.parse_args()
    
    μ_array = np.array(args.mu)
    λ_array = np.array(args.Lambda)
    n1, n2 = args.n1, args.n2
    γ = bo.generate_symlog_points(n1, n2, 0.1, 50, 1)
    n_array = np.array(args.n_array)
    p_array = np.unique((γ * n_array).astype(int))
    snr_array = np.linspace(args.snr[0], args.snr[1], args.snr[2])
    σ = args.sigma
    seed = args.seed

    print('Running Simulations')
    print(f'μ_array: {μ_array}')
    print(f'λ_array: {λ_array}')
    print(f'n_array: {n_array}')
    print(f'p_array: {p_array}')
    print(f'snr_array: {snr_array}')
    print(f'σ: {σ}')
    print(f'seed: {seed}')

    params = μ_array, λ_array, n_array, p_array, snr_array, σ
    bo.run_func_parameters(bo.simulations_lambda_mu, params, 
                        ['λ', 'μ', 'p', 'n', 'snr', 'MSE'], 
                        seed=seed, name='lambda_mu')
    
if __name__ == '__main__':
    run_simulations_lambda_mu(parser)