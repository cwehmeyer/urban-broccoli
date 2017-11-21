import numpy as np
from .laplacian import create_laplacian
from .jacobi import jacobi
from .gaussseidel import gaussseidel
from .sor import sor

def solve(
    rho, h, method='sor', epsilon=1.0, maxiter=10000, maxerr=1e-15, w=None):
    if method.lower() == 'laplacian':
        laplacian = create_laplacian(rho.shape, h)
        phi = np.linalg.solve(laplacian, -rho.reshape(-1,) / epsilon)
        return phi.reshape(rho.shape)
    elif method.lower() == 'jacobi':
        return jacobi(
            rho, h, epsilon=epsilon, maxiter=maxiter, maxerr=maxerr)
    elif method.lower() == 'gauss-seidel':
        return gaussseidel(
            rho, h, epsilon=epsilon, maxiter=maxiter, maxerr=maxerr)
    elif method.lower() == 'sor':
        return sor(
            rho, h, epsilon=epsilon, maxiter=maxiter, maxerr=maxerr, w=w)
    else:
        raise ValueError('unknown method: ' + str(method))
