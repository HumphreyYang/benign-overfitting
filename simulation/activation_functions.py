import numpy as np
import numba

@numba.njit(cache=True, fastmath=True, nogil=True)
def phi_abs(t):
    """
    Purely nonlinear activation function used in Hastie et al. (2022).

    Formula:
    $\varphi_{\mathrm{abs}}(t)=a(|t|-b)$

    where $a=\sqrt{\pi /(\pi-2)} \text { and } b=\sqrt{2 / \pi}$.
 
    Returns
    -------
    phi_abs : function
        Absolute value activation function.
    """

    a = np.sqrt(np.pi / (np.pi - 2))
    b = np.sqrt(2 / np.pi)
    return a * (np.abs(t) - b)

@numba.njit(cache=True, fastmath=True, nogil=True)
def phi_ReLU(t):
    """
    Rectified linear unit activation function.

    Formula:
    $\varphi_{\mathrm{ReLU}}(t)=\max (0, t)$

    Returns
    -------
    phi_ReLU : function
        Rectified linear unit activation function.
    """

    return np.maximum(0, t)

@numba.njit(cache=True, fastmath=True, nogil=True)
def phi_tanh(t):
    """
    Hyperbolic tangent activation function.

    Formula:
    $\varphi_{\tanh }(t)=\tanh (t)$

    Returns
    -------
    phi_tanh : function
        Hyperbolic tangent activation function.
    """

    return np.tanh(t)

@numba.njit(cache=True, fastmath=True, nogil=True)
def phi_gaussian(t):
    """
    Gaussian activation function.

    Formula:
    $\varphi_{\mathrm{Gauss}}(t)=\exp \left(-t^{2}\right)$

    Returns
    -------
    phi_gaussian : function
        Gaussian activation function.
    """

    return np.exp(-t**2)
