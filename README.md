# Benign Overfitting in Linear Regression with Separable Variance–Covariance

This repo contains simulation files for the project **Benign Overfitting in Linear Regression with Separable Variance–Covariance**.

## File Structure

### Simulation Results

The simulation results are in the folder `results`. For large simulation records with size over 100MB, we include a zip file for the results. To unzip the file, run

### Interative Plots

To see the plots interactively, please run the `plot.ipynb` in `visualization` folder using `jupyter notebook` or `jupyter lab`. 

[Jupyter Widgets](https://ipywidgets.readthedocs.io/en/7.x/user_install.html) need to be activated in the jupyter lab environment.

### Simulation Code

Simulations are written in R and Python. The Python version uses Just-In-Time Compilor to speed up the simulation. The R version, although uses paralleled programming, is much slower than the Python version. Both code are in the folder `simulation`.

To run the simulation, change the parameter in the code and run

```bash
python simulation/benign_overfitting.py
```

`fastmath`  is used in the simulation code. To benefit from `fastmath`, please run

```bash
conda install -c numba icc_rt
```

(Read more here about [SVML](https://numba.pydata.org/numba-doc/latest/user/performance-tips.html#intel-svml))

R version is also available but it was 15 times slower than the JIT Python version. To run the R version, run

```bash
Rscript simulation/benign_overfitting.R
```

### Tests

A small number of tests are written to test the simulation code. To run the tests, run

```bash
python -m unittest simulation/tests/test_benign_overfitting.py
```