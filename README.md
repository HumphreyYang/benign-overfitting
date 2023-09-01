# Benign Overfitting in Linear Regression with Separable Variance–Covariance

This repo contains simulation files for the project **Benign Overfitting in Linear Regression with Separable Variance–Covariance** led by Professor Yanrong Yang and Professor Hanlin Shang.

## File Structure

### Simulation Results

The simulation results are in the folder `results`. For large simulation records with size over 100MB, we cut this into two parts. The first part is named as `results[time]_1` and the second part is named as `results[time]_2`.

### Interative Plots

To see the plots interactively, please run the `plot.ipynb` in `visualization` folder using `jupyter notebook` or `jupyter lab`.

### Simulation Code

Simulations are written in R and Python. The Python version uses Just-In-Time Compilor to speed up the simulation. The R version, although uses paralleled programming, is much slower than the Python version. Both code are in the folder `simulation`.

To run the simulation, change the parameter in the code and run
```bash
python simulation/benign_overfitting.py
```


