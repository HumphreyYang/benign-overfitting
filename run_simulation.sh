#!/bin/bash

python simulation/lambda_mu_simulations.py --mu 1 3 5 10 50 100 200 500 --Lambda 1 --n1 30 --n2 30 --n_array 200 --snr 1 5 4 --sigma 1.0 --test_n 10000 --seed 1655
python simulation/lambda_mu_simulations.py --mu 1 --Lambda 1 3 5 10 50 100 200 500 --n1 30 --n2 30 --n_array 200 --snr 1 5 4 --sigma 1.0 --test_n 10000 --seed 1858

python simulation/lambda_mu_simulations.py --mu 1 3 5 10 50 100 200 500 --Lambda 1 --n1 30 --n2 30 --n_array 200 --snr 1 5 4 --sigma 1.0 --test_n 10000 --activation abs --seed 2025
python simulation/lambda_mu_simulations.py --mu 1 3 5 10 50 100 200 500 --Lambda 1 --n1 30 --n2 30 --n_array 200 --snr 1 5 4 --sigma 1.0 --test_n 10000 --activation ReLU --seed 2025
python simulation/lambda_mu_simulations.py --mu 1 3 5 10 50 100 200 500 --Lambda 1 --n1 30 --n2 30 --n_array 200 --snr 1 5 4 --sigma 1.0 --test_n 10000 --activation tanh --seed 2025
python simulation/lambda_mu_simulations.py --mu 1 3 5 10 50 100 200 500 --Lambda 1 --n1 30 --n2 30 --n_array 200 --snr 1 5 4 --sigma 1.0 --test_n 10000 --activation gaussian --seed 2025

python simulation/lambda_mu_simulations.py --mu 1 --Lambda 1 3 5 10 50 100 200 500 --n1 30 --n2 30 --n_array 200 --snr 1 5 4 --sigma 1.0 --test_n 10000 --activation abs --seed 2025
python simulation/lambda_mu_simulations.py --mu 1 --Lambda 1 3 5 10 50 100 200 500 --n1 30 --n2 30 --n_array 200 --snr 1 5 4 --sigma 1.0 --test_n 10000 --activation ReLU --seed 2025
python simulation/lambda_mu_simulations.py --mu 1 --Lambda 1 3 5 10 50 100 200 500 --n1 30 --n2 30 --n_array 200 --snr 1 5 4 --sigma 1.0 --test_n 10000 --activation tanh --seed 2025
python simulation/lambda_mu_simulations.py --mu 1 --Lambda 1 3 5 10 50 100 200 500 --n1 30 --n2 30 --n_array 200 --snr 1 5 4 --sigma 1.0 --test_n 10000 --activation gaussian --seed 2025