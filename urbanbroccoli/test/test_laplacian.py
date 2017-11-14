import numpy as np
import numpy.testing as npt

from ..laplacian import create_laplacian

def test_create_laplacian_1d_2():
    npt.assert_equal(
        create_laplacian(2),
        np.asarray([[-2.0, 2.0], [2.0, -2.0]], dtype=np.float64))

def test_create_laplacian_1d_3():
    npt.assert_equal(
        create_laplacian(3),
        np.asarray([
            [-2.0, 1.0, 1.0],
            [1.0, -2.0, 1.0],
            [1.0, 1.0, -2.0]], dtype=np.float64))

def test_create_laplacian_1d_4():
    npt.assert_equal(
        create_laplacian(4),
        np.asarray([
            [-2.0, 1.0, 0.0, 1.0],
            [1.0, -2.0, 1.0, 0.0],
            [0.0, 1.0, -2.0, 1.0],
            [1.0, 0.0, 1.0, -2.0]], dtype=np.float64))
