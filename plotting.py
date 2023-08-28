import matplotlib.pyplot as plt
import numpy as np

def plot_line_graphs(MSE_matrix, μ_array, λ_array, p_array, n_array):
    
    # Calculate γ values (p/n)
    γ_array = p_array / n_array[:, None]

    # Fix μ and λ, plot relationship with γ
    fixed_μ, fixed_λ = μ_array[0], λ_array[0]
    plt.figure(figsize=(10, 6))
    for i, n in enumerate(n_array):
        plt.plot(γ_array[i, :], MSE_matrix[0, 0, i, :], label=f'n={n}')
    plt.title(f'Fixed μ = {fixed_μ}, λ = {fixed_λ}')
    plt.xlabel('γ (p/n)')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    
    # Fix μ and γ, plot relationship with λ
    fixed_μ, fixed_γ = μ_array[0], γ_array[0, 0]
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

if __name__ == "__main__":
    μ_array = np.linspace(1, 100, 5)
    λ_array = np.linspace(1, 100, 5)
    γ = np.linspace(0.5, 50, 5)
    n_array = np.arange(10, 51, 10)
    p_array = (γ * n_array).astype(int)

    MSE_matrix = np.load('benign-overfitting/mse_matrix.npy')
    
    plot_line_graphs(MSE_matrix, μ_array, λ_array, p_array, n_array)
