import matplotlib.pyplot as plt
import numpy as np

def plot_line_graphs(MSE_matrix, μ_array, λ_array, p_array, n_array):
    
    # Calculate γ values (p/n)
    γ_array = p_array / n_array[:, None]

    # Fix μ and λ, plot relationship with γ
    fixed_μ, fixed_λ = μ_array[-1], λ_array[-1]
    plt.figure(figsize=(10, 6))
    for i, n in enumerate(n_array):
        plt.plot(γ_array[i, :], MSE_matrix[0, 0, i, :], label=f'n={n}')
    plt.title(f'Fixed μ = {fixed_μ}, λ = {fixed_λ}')
    plt.xlabel('γ (p/n)')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    
    # Fix μ and γ, plot relationship with λ
    fixed_μ, fixed_γ = μ_array[0], γ_array[-1, -1]
    plt.figure(figsize=(10, 6))
    for i, n in enumerate(n_array):
        plt.plot(λ_array, MSE_matrix[0, :, i, 0], label=f'n={n}')
    plt.title(f'Fixed μ = {fixed_μ}, γ = {fixed_γ}')
    plt.xlabel('λ')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    
    # Fix λ and γ, plot relationship with μ
    fixed_λ, fixed_γ = λ_array[0], γ_array[0, 0]
    plt.figure(figsize=(10, 6))
    for i, n in enumerate(n_array):
        plt.plot(μ_array, MSE_matrix[:, 0, i, 0], label=f'n={n}')
    plt.title(f'Fixed λ = {fixed_λ}, γ = {fixed_γ}')
    plt.xlabel('μ')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

def find_nearest_index(array, value):
    """
    Given an array and a value, find the index of the nearest element.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def plot_MSE_against_parameters(MSE_matrix, μ_array, λ_array, n_array, p_array, γ, fixed_μ=None, fixed_λ=None, fixed_γ=None):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    if fixed_μ is not None:
        idx_μ = find_nearest_index(μ_array, fixed_μ)
    else:
        idx_μ = 0  

    if fixed_λ is not None:
        idx_λ = find_nearest_index(λ_array, fixed_λ)
    else:
        idx_λ = 0  

    if fixed_γ is not None:
        idx_γ = find_nearest_index(γ, fixed_γ)
    else:
        idx_γ = 0  

    # MSE against λ for fixed μ and γ
    axs[0].plot(λ_array, MSE_matrix[idx_μ, :, 0, idx_γ])
    axs[0].plot(λ_array, MSE_matrix[idx_μ, :, 0, idx_γ], 'o')
    axs[0].set_title(f'MSE against λ for μ = {μ_array[idx_μ]} and γ = {γ[idx_γ]}')
    axs[0].set_xlabel('λ')
    axs[0].set_ylabel('MSE')

    # MSE against μ for fixed λ and γ
    axs[1].plot(μ_array, MSE_matrix[:, idx_λ, 0, idx_γ])
    axs[1].plot(μ_array, MSE_matrix[:, idx_λ, 0, idx_γ], 'o')
    axs[1].set_title(f'MSE against μ for λ = {λ_array[idx_λ]} and γ = {γ[idx_γ]}')
    axs[1].set_xlabel('μ')
    axs[1].set_ylabel('MSE')

    # MSE against γ for fixed λ and μ
    axs[2].plot(γ, MSE_matrix[idx_μ, idx_λ, 0, :])
    axs[2].plot(γ, MSE_matrix[idx_μ, idx_λ, 0, :], 'o')
    axs[2].set_title(f'MSE against γ for λ = {λ_array[idx_λ]} and μ = {μ_array[idx_μ]}')
    axs[2].set_xlabel('γ (p/n)')
    axs[2].set_ylabel('MSE')

    plt.tight_layout()
    plt.savefig('MSE_against_parameters.png')
    plt.show()

from mpl_toolkits.mplot3d import Axes3D

def plot_3D_MSE(MSE_matrix, μ_array, λ_array, γ_array, n_array, fixed_μ=None, fixed_λ=None, fixed_γ=None):
    idx_n = 0  # Assuming you're working with the first element in n_array for this example
    
    # Generate coordinate matrices
    if fixed_μ is not None:
        idx_μ = find_nearest_index(μ_array, fixed_μ)
        λ_grid, γ_grid = np.meshgrid(λ_array, γ_array)
        Z = MSE_matrix[idx_μ, :, idx_n, :]
        
        x_label, y_label = 'λ', 'γ (p/n)'
        X, Y = λ_grid, γ_grid
        
    elif fixed_λ is not None:
        idx_λ = find_nearest_index(λ_array, fixed_λ)
        μ_grid, γ_grid = np.meshgrid(μ_array, γ_array)
        Z = MSE_matrix[:, idx_λ, idx_n, :]
        
        x_label, y_label = 'μ', 'γ (p/n)'
        X, Y = μ_grid, γ_grid
        
    elif fixed_γ is not None:
        idx_γ = find_nearest_index(γ_array, fixed_γ)
        μ_grid, λ_grid = np.meshgrid(μ_array, λ_array)
        Z = MSE_matrix[:, :, idx_n, idx_γ]
        
        x_label, y_label = 'μ', 'λ'
        X, Y = μ_grid, λ_grid
    
    # Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel('MSE')
    plt.savefig('MSE_3d.png')
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    μ_param = (1, 100, 5)
    λ_param = (1, 100, 5)
    γ_param = (0.5, 15, 20)
    n_param = [200]
    snr_param = 5

    μ_array = np.linspace(*μ_param)
    λ_array = np.linspace(*λ_param)
    γ = np.linspace(*γ_param)
    n_array = np.array(n_param)
    p_array = (γ * n_array).astype(int)
    snr = snr_param


    MSE_matrix = np.load(f'mse_matrix_{μ_param}_{λ_param}_{γ_param}_{n_param}_{snr_param}.npy')
    
    plot_MSE_against_parameters(MSE_matrix, μ_array, λ_array, n_array, p_array, γ, fixed_μ=1, fixed_λ=1, fixed_γ=100)
    plot_3D_MSE(MSE_matrix, μ_array, λ_array, γ, n_array, fixed_γ=1)