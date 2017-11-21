import numpy as np
import numpy.testing as npt
from ..gaussseidel import gaussseidel_1d
from ..gaussseidel import gaussseidel_2d
from ..gaussseidel import gaussseidel_3d
from ..laplacian import create_laplacian

def test_gaussseidel_1d_random():
    x = np.linspace(-np.pi, np.pi, 101)
    rho = np.sin(x)
    phi = gaussseidel_1d(rho, x[1] - x[0], 1.0, 10000, 1e-15)
    laplacian = create_laplacian(rho.shape, [x[1] - x[0]])
    npt.assert_allclose(-np.dot(laplacian, phi), rho, atol=0.001)

def test_gaussseidel_2d_sine():
    gx = np.linspace(-np.pi, np.pi, 25)
    gy = np.linspace(-np.pi, np.pi, 24)
    x, y = np.meshgrid(gx, gy)
    rho = np.sin(x) + 2.0 * np.sin(y)
    h = [gx[1] - gx[0], gy[1] - gy[0]]
    phi = gaussseidel_2d(rho, *h, 1.0, 10000, 1e-15)
    laplacian = create_laplacian(rho.shape, h)
    npt.assert_allclose(
        -np.dot(laplacian, phi.reshape(-1)).reshape(rho.shape),
        rho, atol=0.001)

def test_gaussseidel_3d_sine():
    gx = np.linspace(-np.pi, np.pi, 11)
    gy = np.linspace(-np.pi, np.pi, 10)
    gz = np.linspace(-np.pi, np.pi, 9)
    x, y, z = np.meshgrid(gx, gy, gz)
    rho = np.sin(x) + 2.0 * np.sin(y) + 3.0 * np.sin(z)
    h = [gx[1] - gx[0], gy[1] - gy[0], gz[1] - gz[0]]
    phi = gaussseidel_3d(rho, *h, 1.0, 10000, 1e-15)
    laplacian = create_laplacian(rho.shape, h)
    npt.assert_allclose(
        -np.dot(laplacian, phi.reshape(-1)).reshape(rho.shape),
        rho, atol=0.001)
