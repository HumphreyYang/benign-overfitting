#!/bin/bash

python simulation/simulations.py --mu 1 10 25 50 100 200 500 --Lambda 1 --n1 30 --n2 30 --n_array 200 --snr 1 5 4 --sigma 1.0 --seed 1505
python simulation/simulations.py --mu 1 --Lambda 1 10 25 50 100 200 500 --n1 30 --n2 300 --n_array 200 --snr 1 5 4 --sigma 1.0 --seed 1505
