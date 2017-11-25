import numpy as np

def gaussseidel_1d(rho, hx, epsilon, maxiter, maxerr):
    if rho.ndim != 1:
        raise ValueError("rho must be of shape=(n,)")
    phi = np.zeros(shape=rho.shape, dtype=rho.dtype)
    nx = rho.shape[0]
    mr = hx * hx / epsilon
    for iteration in range(maxiter):
        error = 0.0
        for x in range(nx):
            phi_x = (
                phi[(x - 1) % nx] + \
                phi[(x + 1) % nx] + \
                rho[x] * mr) / 2.0
            error += (phi[x] - phi_x)**2
            phi[x] = phi_x
        if error < maxerr:
            break
    return phi

def gaussseidel_2d(rho, hx, hy, epsilon, maxiter, maxerr):
    if rho.ndim != 2:
        raise ValueError("rho must be of shape=(nx, ny)")
    phi = np.zeros(shape=rho.shape, dtype=rho.dtype)
    nx, ny = rho.shape
    fx, fy = 1.0 / (hx * hx), 1.0 / (hy * hy)
    mx = 0.5 * fx / (fx + fy)
    my = 0.5 * fy / (fx + fy)
    mr = 0.5 / (epsilon * (fx + fy))
    for iteration in range(maxiter):
        error = 0.0
        for x in range(nx):
            for y in range(ny):
                phi_xy = (
                    phi[(x - 1) % nx, y] * mx + \
                    phi[(x + 1) % nx, y] * mx + \
                    phi[x, (y - 1) % ny] * my + \
                    phi[x, (y + 1) % ny] * my + \
                    rho[x, y] * mr)
                error += (phi[x, y] - phi_xy)**2
                phi[x, y] = phi_xy
        if error < maxerr:
            break
    return phi

def gaussseidel_3d(rho, hx, hy, hz, epsilon, maxiter, maxerr):
    if rho.ndim != 3:
        raise ValueError("rho must be of shape=(nx, ny, nz)")
    phi = np.zeros(shape=rho.shape, dtype=rho.dtype)
    nx, ny, nz = rho.shape
    fx, fy, fz = 1.0 / (hx * hx), 1.0 / (hy * hy), 1.0 / (hz * hz)
    mx = 0.5 * fx / (fx + fy + fz)
    my = 0.5 * fy / (fx + fy + fz)
    mz = 0.5 * fz / (fx + fy + fz)
    mr = 0.5 / (epsilon * (fx + fy + fz))
    for iteration in range(maxiter):
        error = 0.0
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    phi_xyz = (
                        phi[(x - 1) % nx, y, z] * mx + \
                        phi[(x + 1) % nx, y, z] * mx + \
                        phi[x, (y - 1) % ny, z] * my + \
                        phi[x, (y + 1) % ny, z] * my + \
                        phi[x, y, (z - 1) % nz] * mz + \
                        phi[x, y, (z + 1) % nz] * mz + \
                        rho[x, y, z] * mr)
                    error += (phi[x, y, z] - phi_xyz)**2
                    phi[x, y, z] = phi_xyz
        if error < maxerr:
            break
    return phi

def gaussseidel(rho, h, epsilon=1.0, maxiter=10000, maxerr=1e-15):
    '''
    Gauss-Seidel relaxation method to solve a linear system of equation
    =====
    
    This function checks the dimension of the function rho passed and calls the right gaussseidel according to it.
    If the dimension is not 1, 2 or 3, an exception is raised.
    
    Arguments:
        rho (ndarray): the density function
        h (ndarray): grid spacing
        epsilon (float): permittivity constant
        maxiter (int): the maximum number of iterations
        maxerr (float): the maximum accepted error, if reached, the iteration stops
        
    Returns:
        the solution given by the gausseilde_Xd function called (where X is the dimension of rho) 
    '''
    
    #Checks the dimension of rho, and calls a function to calculate it accordingly to its dimension
    if rho.ndim == 1:
        
        #rho's dimension is 1
        return gaussseidel_1d(rho, *h, epsilon, maxiter, maxerr)
    elif rho.ndim == 2:
        
        #rho's dimension is 2
        return gaussseidel_2d(rho, *h, epsilon, maxiter, maxerr)
    elif rho.ndim == 3:
        
        #rho's dimension is 3
        return gaussseidel_3d(rho, *h, epsilon, maxiter, maxerr)
    
    #a rho with a non-accettable dimension was passed
    else:
        raise ValueError(
            'gaussseidel expects rho with 1, 2 or 3 dimensions.')
