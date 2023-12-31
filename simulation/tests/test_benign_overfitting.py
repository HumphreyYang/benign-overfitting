import unittest
import numpy as np
import simulation.benign_overfitting as bo 

seed = np.random.randint(0, 2**32 - 1)
print('seed =', seed)

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

def is_pos_semidef(X, ϵ=1e-5):
    return np.all(np.linalg.eigvals(X) >= -ϵ)

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
        β_hat = bo.solve_β_hat(X, Y)
        np.testing.assert_allclose(β_hat, β_hat_expected, rtol=1e-8)
        
        # Test with ill-conditioned matrix
        X = np.array([[1.0, 2.0], [1.0, 2.0001], [1, 2.0002]])
        Y = np.array([3.0, 3.0001, 3.0002])
        β_hat = bo.solve_β_hat(X, Y)
        β_hat_expected = np.linalg.lstsq(X, Y, rcond=None)[0]
        np.testing.assert_allclose(β_hat, β_hat_expected, rtol=1e-5)

    def test_calculate_MSE(self):
        # Test with zero error
        X = np.array([[1.0], [2.0], [3.0]])
        β_hat = np.array([2.0])
        β = np.array([2.0])

        self.assertAlmostEqual(bo.calculate_MSE(β_hat, β, X), 0, delta=1e-8)
        
        # Test with non-zero error
        β_hat = np.array([3.0])

        self.assertAlmostEqual(bo.calculate_MSE(β_hat, β, X), 14/3, delta=1e-8)

    def test_compute_Y(self):
        # Test with zero noise
        X = np.array([[1.0], [1.0], [1.0]])
        β = np.array([1.0])
        σ = 0.0
        Y_expected = np.array([1.0, 1.0, 1.0])
        Y_actual = bo.compute_Y(X, β, σ)
        np.testing.assert_allclose(Y_actual, Y_expected, rtol=1e-8)
        
        # Test with non-zero noise
        σ = 1.0
        np.random.seed(seed)
        noise = np.random.normal(0, σ, len(X))
        Y_expected = X @ β + noise
        Y_actual = bo.compute_Y(X, β, noise)
        np.testing.assert_allclose(Y_actual, Y_expected, rtol=1e-8)

    def test_compute_X(self):
        λ = 1
        μ = 1
        p = 2
        n = 3

        U = bo.generate_orthonormal_matrix(p)
        V = bo.generate_orthonormal_matrix(n)

        np.random.seed(seed)
        Z_expected = np.random.normal(0.0, 1.0, (n, p))
        Γ = V @ np.diag([μ] + [1] * (n-1)) @ V.T
        C = U @ np.diag([λ] + [1] * (p-1)) @ U.T

        self.assertTrue(is_pos_semidef(Γ))
        self.assertTrue(is_pos_semidef(C))

        X_expected = V @ np.diag([μ] + [1] * (n-1)) @ V.T @ Z_expected @ U @ np.diag([λ] + [1] * (p-1)) @ U.T
        X_actual = bo.compute_X(λ, μ, n, p, seed=seed)
        np.testing.assert_allclose(X_actual, X_expected, rtol=1e-8)

    def test_scale_norm(self):
        # Test with unit vector
        X = np.array([1.0, 0.0, 0.0])
        out_norm = 2
        X_scaled = bo.scale_norm(X, out_norm)
        self.assertAlmostEqual(np.linalg.norm(X_scaled)**2, out_norm, delta=1e-8)
        
        # Test with zero vector
        X = np.array([0.0, 0.0, 0.0])
        X_scaled = bo.scale_norm(X, out_norm)
        self.assertAlmostEqual(np.linalg.norm(X_scaled), 0, delta=1e-8)

    def test_generate_orthonormal_matrix(self):

        for dim in range(1, 10):
            A = bo.generate_orthonormal_matrix(dim)
            self.assertTrue(check_orthonormal(A))

    if __name__ == '__main__':
        unittest.main()
