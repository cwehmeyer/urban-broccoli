import numpy as np
import numpy.testing as npt
from ..api import solve
from ..laplacian import create_laplacian

def test_solve_consistency_1d():
    rho = np.random.rand(60)
    rho -= np.mean(rho)
    h = [0.1 + 0.9 * np.random.rand(1)]
    laplacian = create_laplacian(rho.shape, h)
    epsilon = 0.7 + 0.3 * np.random.rand()
    for method in ['laplacian', 'jacobi', 'gauss-seidel', 'sor']:
        phi = solve(rho, h, method=method, epsilon=epsilon)
        npt.assert_allclose(
            -np.dot(laplacian, phi) * epsilon,
            rho, atol=0.2)

def test_solve_consistency_2d():
    rho = np.random.rand(15, 15)
    rho -= np.mean(rho)
    h = 0.1 + 0.9 * np.random.rand(2)
    laplacian = create_laplacian(rho.shape, h)
    epsilon = 0.7 + 0.3 * np.random.rand()
    for method in ['laplacian', 'jacobi', 'gauss-seidel', 'sor']:
        phi = solve(rho, h, method=method, epsilon=epsilon)
        npt.assert_allclose(
            -np.dot(laplacian, phi.reshape(-1)).reshape(rho.shape) * epsilon,
            rho, atol=0.2)

def test_solve_consistency_3d():
    rho = np.random.rand(7, 7, 7)
    rho -= np.mean(rho)
    h = 0.1 + 0.9 * np.random.rand(3)
    laplacian = create_laplacian(rho.shape, h)
    epsilon = 0.7 + 0.3 * np.random.rand()
    for method in ['laplacian', 'jacobi', 'gauss-seidel', 'sor']:
        phi = solve(rho, h, method=method, epsilon=epsilon)
        npt.assert_allclose(
            -np.dot(laplacian, phi.reshape(-1)).reshape(rho.shape) * epsilon,
            rho, atol=0.2)
