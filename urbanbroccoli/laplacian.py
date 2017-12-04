from numpy import zeros, float64

''' Generates an laplace sparse matrix of an 1D-vector.

Arguments:
    nx (integer): length of the vector
    hx (integer): length of the grid in x:x

Return:
    laplacian sparse matrix
'''
def create_laplacian_1d(nx, hx):
    laplacian = zeros(shape=(nx, nx), dtype=float64)
    mx = 1.0 / (hx * hx)
    for x in range(nx):
        laplacian[x, x] = -2.0 * mx
        laplacian[x, (x + 1) % nx] += mx
        laplacian[x, (x - 1) % nx] += mx
    return laplacian

def create_laplacian_2d(nx, ny, hx, hy):
    laplacian = zeros(shape=(nx, ny, nx, ny), dtype=float64)
    mx = 1.0 / (hx * hx)
    my = 1.0 / (hy * hy)
    for x in range(nx):
        for y in range(ny):
            laplacian[x, y, x, y] = -2.0 * (mx + my)
            laplacian[x, y, (x + 1) % nx, y] += mx
            laplacian[x, y, (x - 1) % nx, y] += mx
            laplacian[x, y, x, (y + 1) % ny] += my
            laplacian[x, y, x, (y - 1) % ny] += my
    return laplacian.reshape(nx * ny, nx * ny)

def create_laplacian_3d(nx, ny, nz, hx, hy, hz):
    laplacian = zeros(shape=(nx, ny, nz, nx, ny, nz), dtype=float64)
    mx = 1.0 / (hx * hx)
    my = 1.0 / (hy * hy)
    mz = 1.0 / (hz * hz)
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                laplacian[x, y, z, x, y, z] = -2.0 * (mx + my + mz)
                laplacian[x, y, z, (x + 1) % nx, y, z] += mx
                laplacian[x, y, z, (x - 1) % nx, y, z] += mx
                laplacian[x, y, z, x, (y + 1) % ny, z] += my
                laplacian[x, y, z, x, (y - 1) % ny, z] += my
                laplacian[x, y, z, x, y, (z + 1) % nz] += mz
                laplacian[x, y, z, x, y, (z - 1) % nz] += mz
    return laplacian.reshape(nx * ny * nz, nx * ny * nz)

''' Generates an laplace sparse matrix of an 1D, 2D or 3D input. Delegates the task to the respective function.

Arguments:
    n (array): vector with 1-3 Dimensions
    h (array): length of the grid in 1-3 Dimensions

Return:
    laplacian sparse matrix
'''
def create_laplacian(n, h):
    if len(n) == 1:
        return create_laplacian_1d(*n, *h)
    elif len(n) == 2:
        return create_laplacian_2d(*n, *h)
    elif len(n) == 3:
        return create_laplacian_3d(*n, *h)
    else:
        raise ValueError(
            'create_laplacian expects 1, 2 or 3 positive integers.')
