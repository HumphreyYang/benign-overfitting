import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from ipywidgets import interactive
from IPython.display import display
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interactive, widgets
import pandas as pd

def interative_param_vs_mse(df, fixed_μ, fixed_λ, fixed_γ):
    fig, axs = plt.subplots(3, 1, figsize=(12, 16))
    plot_param_vs_mse(df, 'γ', {'μ': fixed_μ, 'λ': fixed_λ}, axs[0])
    plot_param_vs_mse(df, 'λ', {'μ': fixed_μ, 'γ': fixed_γ}, axs[1])
    plot_param_vs_mse(df, 'μ', {'λ': fixed_λ, 'γ': fixed_γ}, axs[2])
    plt.savefig('MSE_against_parameters.png')
    plt.tight_layout()
    plt.show()

def plot_param_vs_mse(data, param_to_vary, fixed_params, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    nearest_fixed_params = {param: find_nearest_value(data[param].unique(), value) for param, value in fixed_params.items()}
    
    filter_conditions = np.all([data[param] == nearest_value for param, nearest_value in nearest_fixed_params.items()], axis=0)
    filtered_data = data[filter_conditions]

    print(len(filtered_data))

    ax.plot(filtered_data[param_to_vary], filtered_data['MSE'], marker='o')
    ax.set_xlabel(param_to_vary)
    ax.set_ylabel('MSE')
    ax.set_title(f'{param_to_vary} vs MSE (Nearest Fixed Params: {list(fixed_params.items())}')
    ax.grid(True)
    return ax

def interative_bars_line(df):
    # Create sliders
    μ_slider = widgets.FloatSlider(min=np.min(df['μ']), max=np.max(df['μ']), step=(np.max(df['μ'])-np.min(df['μ']))/50, description='μ')
    λ_slider = widgets.FloatSlider(min=np.min(df['λ']), max=np.max(df['λ']), step=(np.max(df['λ'])-np.min(df['λ']))/50, description='λ')
    γ_slider = widgets.FloatSlider(min=np.min(df['γ']), max=np.max(df['γ']), step=(np.max(df['γ'])-np.min(df['γ']))/50, description='γ')

    # Generate interactive plots
    interactive_plot = interactive(lambda fixed_μ, fixed_λ, fixed_γ: 
                                interative_param_vs_mse(df, fixed_μ=fixed_μ, fixed_λ=fixed_λ, fixed_γ=fixed_γ), 
                                fixed_μ=μ_slider, fixed_λ=λ_slider, fixed_γ=γ_slider)
    return interactive_plot

def interative_bars_3d(df):
    # Create sliders
    x_param = widgets.Dropdown(options=['μ', 'λ', 'γ'], description='x-axis')
    y_param = widgets.Dropdown(options=['λ', 'μ', 'γ'], description='y-axis')
    fixed_params = widgets.Dropdown(options=['γ', 'μ', 'λ'], description='Fixed Params')
    fixed_params_slider = widgets.FloatSlider(min=0, max=100, 
                                              step=100/50, description='value for fixed parameter')

    # Generate interactive plots
    interactive_plot = interactive(lambda x_param, y_param, fixed_params, fixed_params_values: 
                                   plot_surface_MSE(df, x_param=x_param, y_param=y_param, fixed_params=fixed_params, 
                                                    fixed_params_values=fixed_params_values), x_param=x_param, 
                                                    y_param=y_param, fixed_params=fixed_params, fixed_params_values=fixed_params_slider)
    return interactive_plot


def find_nearest_value(array, value):
    return min(array, key=lambda x: abs(x - value))

from scipy.interpolate import griddata

def plot_surface_MSE(data, x_param, y_param, fixed_params, fixed_params_values, z_param='MSE'):
    """
    Generate a surface plot for MSE against varying parameters, given fixed values for other parameters.
    
    Parameters:
    - data: DataFrame containing the data
    - x_param, y_param: Parameters for the x and y axis
    - z_param: Parameter for the z axis, default is 'MSE'
    - fixed_params: Dictionary of fixed parameter values
    """
    # Find the nearest available values in the dataset for the fixed parameters
    nearest_fixed_params = {fixed_params: find_nearest_value(data[fixed_params].unique(), fixed_params_values)}
    
    # Filter the data based on the nearest fixed parameters
    filter_conditions = np.all([data[param] == nearest_value for param, 
                                nearest_value in nearest_fixed_params.items()], axis=0) if nearest_fixed_params else np.array([True]*len(data))
    filtered_data = data[filter_conditions]
    
    # Extract the data for surface plotting
    x = filtered_data[x_param].values
    y = filtered_data[y_param].values
    z = filtered_data[z_param].values

    # Create a grid
    x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 50),
                                 np.linspace(min(y), max(y), 50))
    
    # Interpolate z values for the grid
    z_grid = griddata((x, y), z, (x_grid, y_grid), method='linear')
    
    # Generate surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', edgecolor='k')
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_zlabel(z_param)
    plt.title(f'Surface plot of {z_param} (Fixed Params: {nearest_fixed_params})')
    plt.show()

if __name__ == "__main__":
    file_name = 'results_[30-08-2023_12:11:02].csv'
    df = pd.read_csv(file_name)
    df['γ'] = df['p'] / df['n']

    #Plot for λ vs MSE (fix μ and γ)
    plot_param_vs_mse(df, 'λ', {'μ': 50, 'γ': 2.7})

    # Plot for μ vs MSE (fix λ and γ)
    plot_param_vs_mse(df, 'μ', {'λ': 50, 'γ': 2.7})

    # Plot for γ vs MSE (fix λ and μ)
    plot_param_vs_mse(df, 'γ', {'λ': 50, 'μ': 50})

    plot_surface_MSE(df, 'μ', 'γ', fixed_params={'λ': 100})

    interative_bars_line(df)