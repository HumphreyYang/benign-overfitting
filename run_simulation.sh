#!/bin/bash

python simulation/lambda_mu_simulations.py --mu 1 1.5 2 5 10 50 100 200 500 --Lambda 1 --n1 30 --n2 60 --n_array 20 --snr 1 5 4 --sigma 1.0 --test_n 1000 --seed 1505
python simulation/lambda_mu_simulations.py --mu 1 --Lambda 1 1.5 2 5 10 50 100 200 500 --n1 30 --n2 60 --n_array 20 --snr 1 5 4 --sigma 1.0 --test_n 1000 --seed 1239
