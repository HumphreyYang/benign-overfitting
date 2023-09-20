# Benign Overfitting in Linear Regression with Separable Sample Covariance Matrices

This repo contains simulation files for the project **Benign Overfitting in Linear Regression with Separable Sample Covariance Matrices**.

## Environment Setup

To set up the environment, run

```bash
conda create -n bo python==3.10
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

To run the simulation, change the parameter in the code and run

```bash
python simulation/benign_overfitting_efficient.py
```

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