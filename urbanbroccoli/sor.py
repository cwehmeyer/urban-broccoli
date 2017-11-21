import numpy as np

def sor_1d(rho, hx, epsilon, maxiter, maxerr, w):
    if rho.ndim != 1:
        raise ValueError("rho must be of shape=(n,)")
    phi = np.zeros(shape=rho.shape, dtype=rho.dtype)
    nx = rho.shape[0]
    mr = hx * hx / epsilon
    if w is None:
        w = 2.0 / (1.0 + np.pi / float(nx))
    for iteration in range(maxiter):
        error = 0.0
        for x in range(nx):
            if x % 2 == 0: continue
            phi_x = (
                phi[(x - 1) % nx] + \
                phi[(x + 1) % nx] + \
                rho[x] * mr) / 2.0
            phi[x] = (1.0 - w) * phi[x] + w * phi_x
            error += (phi[x] - phi_x)**2
        for x in range(nx):
            if x % 2 != 0: continue
            phi_x = (
                phi[(x - 1) % nx] + \
                phi[(x + 1) % nx] + \
                rho[x] * mr) / 2.0
            phi[x] = (1.0 - w) * phi[x] + w * phi_x
            error += (phi[x] - phi_x)**2
        if error < maxerr:
            break
    return phi

def sor_2d(rho, hx, hy, epsilon, maxiter, maxerr, w):
    if rho.ndim != 2:
        raise ValueError("rho must be of shape=(nx, ny)")
    phi = np.zeros(shape=rho.shape, dtype=rho.dtype)
    nx, ny = rho.shape
    fx, fy = 1.0 / (hx * hx), 1.0 / (hy * hy)
    mx = 0.5 * fx / (fx + fy)
    my = 0.5 * fy / (fx + fy)
    mr = 0.5 / (epsilon * (fx + fy))
    if w is None:
        w = 2.0 / (1.0 + np.pi / float(0.5 * (nx + ny)))
    for iteration in range(maxiter):
        error = 0.0
        for x in range(nx):
            for y in range(ny):
                if (x + y) % 2 == 0: continue
                phi_xy = (
                    phi[(x - 1) % nx, y] * mx + \
                    phi[(x + 1) % nx, y] * mx + \
                    phi[x, (y - 1) % ny] * my + \
                    phi[x, (y + 1) % ny] * my + \
                    rho[x, y] * mr)
                phi[x, y] = (1.0 - w) * phi[x, y] + w * phi_xy
                error += (phi[x, y] - phi_xy)**2
        for x in range(nx):
            for y in range(ny):
                if (x + y) % 2 != 0: continue
                phi_xy = (
                    phi[(x - 1) % nx, y] * mx + \
                    phi[(x + 1) % nx, y] * mx + \
                    phi[x, (y - 1) % ny] * my + \
                    phi[x, (y + 1) % ny] * my + \
                    rho[x, y] * mr)
                phi[x, y] = (1.0 - w) * phi[x, y] + w * phi_xy
                error += (phi[x, y] - phi_xy)**2
        if error < maxerr:
            break
    return phi

def sor_3d(rho, hx, hy, hz, epsilon, maxiter, maxerr, w):
    if rho.ndim != 3:
        raise ValueError("rho must be of shape=(nx, ny, nz)")
    phi = np.zeros(shape=rho.shape, dtype=rho.dtype)
    nx, ny, nz = rho.shape
    fx, fy, fz = 1.0 / (hx * hx), 1.0 / (hy * hy), 1.0 / (hz * hz)
    mx = 0.5 * fx / (fx + fy + fz)
    my = 0.5 * fy / (fx + fy + fz)
    mz = 0.5 * fz / (fx + fy + fz)
    mr = 0.5 / (epsilon * (fx + fy + fz))
    if w is None:
        w = 2.0 / (1.0 + np.pi / float(0.5 * (nx + ny + nz)))
    for iteration in range(maxiter):
        error = 0.0
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if (x + y + z) % 2 == 0: continue
                    phi_xyz = (
                        phi[(x - 1) % nx, y, z] * mx + \
                        phi[(x + 1) % nx, y, z] * mx + \
                        phi[x, (y - 1) % ny, z] * my + \
                        phi[x, (y + 1) % ny, z] * my + \
                        phi[x, y, (z - 1) % nz] * mz + \
                        phi[x, y, (z + 1) % nz] * mz + \
                        rho[x, y, z] * mr)
                    phi[x, y, z] = (1.0 - w) * phi[x, y, z] + w * phi_xyz
                    error += (phi[x, y, z] - phi_xyz)**2
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if (x + y + z) % 2 != 0: continue
                    phi_xyz = (
                        phi[(x - 1) % nx, y, z] * mx + \
                        phi[(x + 1) % nx, y, z] * mx + \
                        phi[x, (y - 1) % ny, z] * my + \
                        phi[x, (y + 1) % ny, z] * my + \
                        phi[x, y, (z - 1) % nz] * mz + \
                        phi[x, y, (z + 1) % nz] * mz + \
                        rho[x, y, z] * mr)
                    phi[x, y, z] = (1.0 - w) * phi[x, y, z] + w * phi_xyz
                    error += (phi[x, y, z] - phi_xyz)**2
        if error < maxerr:
            break
    return phi

def sor(rho, h, epsilon=1.0, maxiter=10000, maxerr=1e-15, w=None):
    if rho.ndim == 1:
        return sor_1d(rho, *h, epsilon, maxiter, maxerr, w)
    elif rho.ndim == 2:
        return sor_2d(rho, *h, epsilon, maxiter, maxerr, w)
    elif rho.ndim == 3:
        return sor_3d(rho, *h, epsilon, maxiter, maxerr, w)
    else:
        raise ValueError(
            'sor expects rho with 1, 2 or 3 dimensions.')
