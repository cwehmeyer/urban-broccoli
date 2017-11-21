import numpy as np

def jacobi_1d(rho, hx, epsilon, maxiter, maxerr):
    '''solve poissons equation with a 1d jacobi
    
    arguments:
        rho (array like of float): charge density
        hx (float): grid spacing
        epsilion (float): vacuum permeativity
        maxiter (integer): maximum number of iterations
        maxerr (float): convergence criteria
    '''
    if rho.ndim != 1:
        raise ValueError("rho must be of shape=(n,)")
    phi = np.zeros(shape=rho.shape, dtype=rho.dtype)
    phi_ = np.zeros(shape=rho.shape, dtype=rho.dtype)
    nx = rho.shape[0]
    mr = hx * hx / epsilon
    for iteration in range(maxiter):
        for x in range(nx):
            phi_[x] = (
                phi[(x - 1) % nx] + \
                phi[(x + 1) % nx] + \
                rho[x] * mr) / 2.0
        error = np.sum((phi - phi_)**2)
        phi[:] = phi_
        if error < maxerr:
            break
    return phi

def jacobi_2d(rho, hx, hy, epsilon, maxiter, maxerr):
    if rho.ndim != 2:
        raise ValueError("rho must be of shape=(nx, ny)")
    phi = np.zeros(shape=rho.shape, dtype=rho.dtype)
    phi_ = np.zeros(shape=rho.shape, dtype=rho.dtype)
    nx, ny = rho.shape
    fx, fy = 1.0 / (hx * hx), 1.0 / (hy * hy)
    mx = 0.5 * fx / (fx + fy)
    my = 0.5 * fy / (fx + fy)
    mr = 0.5 / (epsilon * (fx + fy))
    for iteration in range(maxiter):
        for x in range(nx):
            for y in range(ny):
                phi_[x, y] = (
                    phi[(x - 1) % nx, y] * mx + \
                    phi[(x + 1) % nx, y] * mx + \
                    phi[x, (y - 1) % ny] * my + \
                    phi[x, (y + 1) % ny] * my + \
                    rho[x, y] * mr)
        error = np.sum((phi - phi_)**2)
        phi[:] = phi_
        if error < maxerr:
            break
    return phi

def jacobi_3d(rho, hx, hy, hz, epsilon, maxiter, maxerr):
    '''Solve 3D discrete Poisson equation using Jacobi method
    
    arguments:
        rho (array-like of float): charge density
        hx, hy, hz (float): grid spacing
        epsilion (float): vacuum permittivity
        maxiter (integer): maximum number of iterations
        maxerr (float): convergence criteria
    '''
    if rho.ndim != 3:
        raise ValueError("rho must be of shape=(nx, ny, nz)")
    phi = np.zeros(shape=rho.shape, dtype=rho.dtype)
    phi_ = np.zeros(shape=rho.shape, dtype=rho.dtype)
    nx, ny, nz = rho.shape
    fx, fy, fz = 1.0 / (hx * hx), 1.0 / (hy * hy), 1.0 / (hz * hz)
    mx = 0.5 * fx / (fx + fy + fz)
    my = 0.5 * fy / (fx + fy + fz)
    mz = 0.5 * fz / (fx + fy + fz)
    mr = 0.5 / (epsilon * (fx + fy + fz))
    for iteration in range(maxiter):
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    phi_[x, y, z] = (
                        phi[(x - 1) % nx, y, z] * mx + \
                        phi[(x + 1) % nx, y, z] * mx + \
                        phi[x, (y - 1) % ny, z] * my + \
                        phi[x, (y + 1) % ny, z] * my + \
                        phi[x, y, (z - 1) % nz] * mz + \
                        phi[x, y, (z + 1) % nz] * mz + \
                        rho[x, y, z] * mr)
        error = np.sum((phi - phi_)**2)
        phi[:] = phi_
        if error < maxerr:
            break
    return phi

def jacobi(rho, h, epsilon=1.0, maxiter=10000, maxerr=1e-15):
    if rho.ndim == 1:
        return jacobi_1d(rho, *h, epsilon, maxiter, maxerr)
    elif rho.ndim == 2:
        return jacobi_2d(rho, *h, epsilon, maxiter, maxerr)
    elif rho.ndim == 3:
        return jacobi_3d(rho, *h, epsilon, maxiter, maxerr)
    else:
        raise ValueError(
            'jacobi expects rho with 1, 2 or 3 dimensions.')
