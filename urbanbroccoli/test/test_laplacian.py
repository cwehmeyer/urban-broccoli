import numpy as np
import numpy.testing as npt
from ..laplacian import create_laplacian_1d
from ..laplacian import create_laplacian_2d
from ..laplacian import create_laplacian_3d
from ..laplacian import create_laplacian

def test_create_laplacian_1d_2():
    reference = np.asarray([
        [-0.5, 0.5],
        [0.5, -0.5]], dtype=np.float64)
    version1 = create_laplacian_1d(2, 2.0)
    version2 = create_laplacian([2], [2.0])
    npt.assert_equal(version1, reference)
    npt.assert_equal(version2, reference)

def test_create_laplacian_1d_3():
    reference = np.asarray([
        [-8.0, 4.0, 4.0],
        [4.0, -8.0, 4.0],
        [4.0, 4.0, -8.0]], dtype=np.float64)
    version1 = create_laplacian_1d(3, 0.5)
    version2 = create_laplacian([3], [0.5])
    npt.assert_equal(version1, reference)
    npt.assert_equal(version2, reference)

def test_create_laplacian_1d_4():
    reference = np.asarray([
        [-2.0, 1.0, 0.0, 1.0],
        [1.0, -2.0, 1.0, 0.0],
        [0.0, 1.0, -2.0, 1.0],
        [1.0, 0.0, 1.0, -2.0]], dtype=np.float64)
    version1 = create_laplacian_1d(4, 1.0)
    version2 = create_laplacian([4], [1.0])
    npt.assert_equal(version1, reference)
    npt.assert_equal(version2, reference)

def test_create_laplacian_2d_2():
    reference = np.asarray([
        [-1.0, 0.5, 0.5, 0.0],
        [0.5, -1.0, 0.0, 0.5],
        [0.5, 0.0, -1.0, 0.5],
        [0.0, 0.5, 0.5, -1.0]], dtype=np.float64)
    version1 = create_laplacian_2d(2, 2, 2.0, 2.0)
    version2 = create_laplacian([2, 2], [2.0, 2.0])
    npt.assert_equal(version1, reference)
    npt.assert_equal(version2, reference)

def test_create_laplacian_3d_2():
    reference = np.asarray([
        [-24.0, 8.0, 8.0, 0.0, 8.0, 0.0, 0.0, 0.0],
        [8.0, -24.0, 0.0, 8.0, 0.0, 8.0, 0.0, 0.0],
        [8.0, 0.0, -24.0, 8.0, 0.0, 0.0, 8.0, 0.0],
        [0.0, 8.0, 8.0, -24.0, 0.0, 0.0, 0.0, 8.0],
        [8.0, 0.0, 0.0, 0.0, -24.0, 8.0, 8.0, 0.0],
        [0.0, 8.0, 0.0, 0.0, 8.0, -24.0, 0.0, 8.0],
        [0.0, 0.0, 8.0, 0.0, 8.0, 0.0, -24.0, 8.0],
        [0.0, 0.0, 0.0, 8.0, 0.0, 8.0, 8.0, -24.0]], dtype=np.float64)
    version1 = create_laplacian_3d(2, 2, 2, 0.5, 0.5, 0.5)
    version2 = create_laplacian([2, 2, 2], [0.5, 0.5, 0.5])
    npt.assert_equal(version1, reference)
    npt.assert_equal(version2, reference)

def test_create_laplacian_1d_sine():
    x = np.linspace(-np.pi, np.pi, 100)
    rho = np.sin(x)
    h = [x[1] - x[0]]
    laplacian = create_laplacian_1d(*rho.shape, *h)
    phi = np.linalg.solve(laplacian, rho)
    npt.assert_almost_equal(np.dot(laplacian, phi), rho)

def test_create_laplacian_2d_sine():
    gx = np.linspace(-np.pi, np.pi, 25)
    gy = np.linspace(-np.pi, np.pi, 21)
    x, y = np.meshgrid(gx, gy)
    rho = np.sin(x) + 2.0 * np.sin(y)
    h = [gx[1] - gx[0], gy[1] - gy[0]]
    laplacian = create_laplacian_2d(*rho.shape, *h)
    phi = np.linalg.solve(laplacian, rho.reshape(-1))
    npt.assert_almost_equal(np.dot(laplacian, phi).reshape(rho.shape), rho)

def test_create_laplacian_3d_sine():
    gx = np.linspace(-np.pi, np.pi, 11)
    gy = np.linspace(-np.pi, np.pi, 10)
    gz = np.linspace(-np.pi, np.pi, 9)
    x, y, z = np.meshgrid(gx, gy, gz)
    rho = np.sin(x) + 2.0 * np.sin(y) + 3.0 * np.sin(z)
    h = [gx[1] - gx[0], gy[1] - gy[0], gz[1] - gz[0]]
    laplacian = create_laplacian_3d(*rho.shape, *h)
    phi = np.linalg.solve(laplacian, rho.reshape(-1))
    npt.assert_almost_equal(np.dot(laplacian, phi).reshape(rho.shape), rho)

def test_create_laplacian_consistency_1d_random():
    b = np.random.rand(*np.random.randint(low=80, high=101, size=1))
    b -= np.mean(b)
    h = [0.1 + 0.9 * np.random.rand(1)]
    laplacian = create_laplacian_1d(*b.shape, *h)
    npt.assert_equal(create_laplacian(b.shape, h), laplacian)

def test_create_laplacian_consistency_2d_random():
    b = np.random.rand(*np.random.randint(low=20, high=26, size=2))
    b -= np.mean(b)
    h = 0.1 + 0.9 * np.random.rand(2)
    laplacian = create_laplacian_2d(*b.shape, *h)
    npt.assert_equal(create_laplacian(b.shape, h), laplacian)

def test_create_laplacian_consistency_3d_random():
    b = np.random.rand(*np.random.randint(low=8, high=11, size=3))
    b -= np.mean(b)
    h = 0.1 + 0.9 * np.random.rand(3)
    laplacian = create_laplacian_3d(*b.shape, *h)
    npt.assert_equal(create_laplacian(b.shape, h), laplacian)
