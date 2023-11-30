#!/bin/bash

python simulation/lambda_mu_simulations.py --mu 1 3 5 10 50 100 200 500 --Lambda 1 --n1 30 --n2 30 --val 200 --var n --snr 1 5 4 --sigma 1.0 --test_n 10000 --seed 16
python simulation/lambda_mu_simulations.py --mu 1 --Lambda 1 3 5 10 50 100 200 500 --n1 30 --n2 30 --val 200 --var n --snr 1 5 4 --sigma 1.0 --test_n 10000 --seed 18

python simulation/lambda_mu_simulations.py --mu 1 3 5 10 50 100 200 500 --Lambda 1 --n1 30 --n2 30 --val 200 --var n --snr 1 5 4 --sigma 1.0 --test_n 10000 --activation abs --seed 2025
python simulation/lambda_mu_simulations.py --mu 1 --Lambda 1 3 5 10 50 100 200 500 --n1 30 --n2 30 --val 200 --var n --snr 1 5 4 --sigma 1.0 --test_n 10000 --activation abs --seed 2025

python simulation/lambda_mu_simulations.py --mu 1 3 5 10 50 100 200 500 --Lambda 1 --n1 30 --n2 30 --val 200 --var n --snr 1 5 4 --sigma 1.0 --test_n 10000 --activation quad --seed 2025
python simulation/lambda_mu_simulations.py --mu 1 --Lambda 1 3 5 10 50 100 200 500 --n1 30 --n2 30 --val 200 --var n --snr 1 5 4 --sigma 1.0 --test_n 10000 --activation quad --seed 2025

python simulation/bias_simulations.py --mu 1 3 5 10 50 100 200 500 --Lambda 1 --n1 30 --n2 30 --val 200 --var n --snr 1 5 4 --sigma 1.0 --test_n 10000 --seed 1655
python simulation/bias_simulations.py --mu 1 --Lambda 1 3 5 10 50 100 200 500 --n1 30 --n2 30 --val 200 --var n --snr 1 5 4 --sigma 1.0 --test_n 10000 --seed 1858

python simulation/ridge_simulations.py --mu 1 3 5 10 50 100 200 500 --Lambda 1 --n1 30 --n2 30 --val 200 --var n --tau 0 9 1 --snr 1 5 4 --sigma 1.0 --test_n 10000 --seed 1655
python simulation/ridge_simulations.py --mu 1 --Lambda 1 3 5 10 50 100 200 500 --n1 30 --n2 30 --val 200 --var n --tau 0 9 1 --snr 1 5 4 --sigma 1.0 --test_n 10000 --seed 1858