import numba
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import FeatureUnion

def phi_poly(X, degree):
    """
    Computes the polynomial activation for multidimensional matrix X.

    Parameters:
    X (ndarray): Input matrix of shape (N, D) where N is the number of data points and D is the dimensionality.
    degree (int): The degree of the polynomial.

    Returns:
    ndarray: The polynomial activation matrix of shape (N, M) where M is the number of polynomial features.
    """

    poly = PolynomialFeatures(degree=degree)
    return poly.fit_transform(X)

def phi_rbf(x, z, sigma):
    """
    Computes the RBF activation for multidimensional matrices x and z.
    
    Parameters:
    x (ndarray): Input matrix of shape (N, D) where N is the number of data points and D is the dimensionality.
    z (ndarray): Center matrix of shape (M, D) where M is the number of centers and D is the dimensionality.
    sigma (float): The standard deviation (spread) of the Gaussian function.
    
    Returns:
    ndarray: The RBF activation matrix of shape (N, M).
    """
    squared_distance = np.sum((x[:, np.newaxis, :] - z[np.newaxis, :, :])**2, axis=-1)
    
    # Compute the RBF activation
    phi_z = np.exp(-squared_distance / (2 * sigma**2))
    
    return phi_z