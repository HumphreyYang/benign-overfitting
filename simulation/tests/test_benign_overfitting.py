import unittest
import numpy as np
from simulation.benign_overfitting import *


def check_orthonormal(A):
    n, m = A.shape
    if n != m:
        return False

    # Check if columns are unit vectors
    col_norms = np.linalg.norm(A, ord=2)
    if not np.allclose(col_norms, 1, atol=1e-8, rtol=1e-8, equal_nan=False):
       return False
    
    # Check orthogonality
    ortho_check = np.dot(A.T, A)
    if not np.allclose(ortho_check, np.eye(n), atol=1e-8, rtol=1e-8, equal_nan=False):
       return False
    return True

class TestSimulationMethods(unittest.TestCase):

    def test_check_orthonormal(self):
        # Test with identity matrix
        I = np.eye(3)
        self.assertTrue(check_orthonormal(I))
        
        # Test with non-square matrix
        non_square = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        self.assertFalse(check_orthonormal(non_square))
        
        # Test with non-orthonormal matrix
        non_ortho = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.assertFalse(check_orthonormal(non_ortho))

    def test_solve_β_hat(self):
        # Test with well-conditioned matrix
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Y = np.array([3.0, 7.0, 11.0])
        β_hat_expected = np.array([1.0, 1.0])
        β_hat = solve_β_hat(X, Y)
        np.testing.assert_allclose(β_hat, β_hat_expected, rtol=1e-8)
        
        # Test with ill-conditioned matrix
        X = np.array([[1.0, 2.0], [1.0, 2.0001], [1, 2.0002]])
        Y = np.array([3.0, 3.0001, 3.0002])
        β_hat = solve_β_hat(X, Y)
        β_hat_expected = np.linalg.lstsq(X, Y, rcond=None)[0]
        np.testing.assert_allclose(β_hat, β_hat_expected, rtol=1e-5)

    def test_calculate_MSE(self):
        # Test with zero error
        X = np.array([[1.0], [2.0], [3.0]])
        Y = np.array([2.0, 4.0, 6.0])
        β_hat = np.array([2.0])
        self.assertAlmostEqual(calculate_MSE(β_hat, X, Y), 0, delta=1e-8)
        
        # Test with non-zero error
        Y = np.array([2.1, 4.1, 5.9])
        self.assertAlmostEqual(calculate_MSE(β_hat, X, Y), 0.01, delta=1e-8)

    def test_compute_Y(self):
        # Test with zero noise
        X = np.array([[1.0], [1.0], [1.0]])
        β = np.array([1.0])
        σ = 0.0
        Y_expected = np.array([1.0, 1.0, 1.0])
        Y_actual = compute_Y(X, β, σ, seed=42)
        np.testing.assert_allclose(Y_actual, Y_expected, rtol=1e-8)
        
        # Test with non-zero noise
        σ = 1.0
        np.random.seed(42)
        noise = np.random.normal(0, σ, len(X))
        Y_expected = X @ β + noise
        Y_actual = compute_Y(X, β, σ, seed=42)
        np.testing.assert_allclose(Y_actual, Y_expected, rtol=1e-8)

    def test_compute_X(self):
        λ = 1
        μ = 1
        p = 2
        n = 3
        U = np.array([
            [1 / np.sqrt(2), -1 / np.sqrt(2)],
            [1 / np.sqrt(2), 1 / np.sqrt(2)]])
        V = np.array([
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0]])
        np.random.seed(42)
        Z_expected = np.random.normal(0.0, 1.0, (n, p))
        X_expected = V @ np.diag([μ] + [1] * (n-1)) @ V.T @ Z_expected @ U @ np.diag([λ] + [1] * (p-1)) @ U.T
        X_actual = compute_X(λ, μ, p, n, U, V, seed=42)
        np.testing.assert_allclose(X_actual, X_expected, rtol=1e-8)

    def test_scale_norm(self):
        # Test with unit vector
        X = np.array([1.0, 0.0, 0.0])
        out_norm = 2
        X_scaled = scale_norm(X, out_norm)
        self.assertAlmostEqual(np.linalg.norm(X_scaled), out_norm, delta=1e-8)
        
        # Test with zero vector
        X = np.array([0.0, 0.0, 0.0])
        X_scaled = scale_norm(X, out_norm)
        self.assertAlmostEqual(np.linalg.norm(X_scaled), 0, delta=1e-8)

    def test_generate_orthonormal_matrix(self):
        # Test for dimensionality
        dim = 3
        A = generate_orthonormal_matrix(dim, seed=42)
        self.assertEqual(A.shape, (dim, dim))
        
        # Test for orthonormality
        self.assertTrue(check_orthonormal(A))
        
        for _ in range(1000):
            B = generate_orthonormal_matrix(dim, seed=42)
            self.assertTrue(check_orthonormal(B))
            np.testing.assert_allclose(A, B, rtol=1e-8)

        B_pre = A.copy()
        for _ in range(10):      
            B = generate_orthonormal_matrix(dim)
            self.assertTrue(check_orthonormal(B))
            np.testing.assert_equal(np.any(np.not_equal(B, B_pre)), True)
            B_pre = B.copy()
    if __name__ == '__main__':
        unittest.main()
