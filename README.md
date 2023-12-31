# Benign Overfitting in Linear Regression with Separable Sample Covariance Matrices

This repo contains simulation files for an ongoing project on **Benign Overfitting in Linear Regression with Separable Sample Covariance Matrices** supervised by Professor Yanrong Yang and Professor Hanlin Shang.

There are delays in the documentation of the project due to active development. We will update the documentation once experiments are finalized.

## Environment Setup

To set up the environment, run

```bash
conda create -n bo python==3.9.16
```

and install the packages in `requirements.txt` using

```bash
pip install -r requirements.txt
```

## File Structure

### Simulation Results

The simulation results are in the folder `results`. For large simulation records with size over 100MB, we include a zip file for the results. To unzip the file, run

### Interative Plots

To see the plots interactively, please run the `plot.ipynb` in `visualization` folder using `jupyter notebook` or `jupyter lab`. 

[Jupyter Widgets](https://ipywidgets.readthedocs.io/en/7.x/user_install.html) need to be activated in the jupyter lab environment.

### Simulation Code

Simulations are written in R and Python. The Python version uses Just-In-Time Compilor to speed up the simulation. The R version is also available. Both code are in the folder `simulation`.

To replicate our simulation, run

```bash
sh run_simulation.sh
```

Parameters can be changed in the file `run_simulation.sh`.

`fastmath`  is used in the simulation code. To benefit from `fastmath`, please run

```bash
conda install -c numba icc_rt
```

(Read more here about [SVML](https://numba.pydata.org/numba-doc/latest/user/performance-tips.html#intel-svml))

### Tests

A small number of tests are written to test the simulation code. To run the tests, run

```bash
python -m unittest simulation/tests/test_benign_overfitting.py
```