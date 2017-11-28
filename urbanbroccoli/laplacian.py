from numpy import zeros, float64

def create_laplacian_1d(nx, hx):
    laplacian = zeros(shape=(nx, nx), dtype=float64)
    mx = 1.0 / (hx * hx)
    for x in range(nx):
        laplacian[x, x] = -2.0 * mx
        laplacian[x, (x + 1) % nx] += mx
        laplacian[x, (x - 1) % nx] += mx
    return laplacian

def create_laplacian_2d(nx, ny, hx, hy):
    '''Generates 2d Laplacian matrix with respect to grid spacing.   
       
    Creates 4d array to show components of laplace operator at each grid point
    and fills it as per discetized laplace operator taking care of
    periodic boundary conditions.  
    In the end reshapes it into 2d laplaction matrix .  
    
    Parameters:
        nx(integer): number of grid points in X-direction
        ny(integer): number of grid points in Y-direction
        hx(float): grid-spacing in X-direction
        hy(float): grid-spacing in Y-direction
    
    Returns:
        Laplacian matrix of size(nx * ny, nx * ny)  
    '''
    laplacian = zeros(shape=(nx, ny, nx, ny), dtype=float64)       
    mx = 1.0 / (hx * hx)
    my = 1.0 / (hy * hy)
    for x in range(nx):
        for y in range(ny):
            laplacian[x, y, x, y] = -2.0 * (mx + my)
            laplacian[x, y, (x + 1) % nx, y] += mx       # sets laplacian component of neighbour grid point in X-direction
            laplacian[x, y, (x - 1) % nx, y] += mx       # sets component of neighbour grid point in opposite X-direction
            laplacian[x, y, x, (y + 1) % ny] += my       # sets component of neighbour grid point in Y-direction
            laplacian[x, y, x, (y - 1) % ny] += my       # sets component of neighbour grid point in opposite Y-direction
    return laplacian.reshape(nx * ny, nx * ny)

def create_laplacian_3d(nx, ny, nz, hx, hy, hz):
    '''Generate a 3d laplacian matrix with respect to various grid sizes.
    
    We generate a laplacian matrix for three dimensions by creating a 6d tensor 
    and filling it according to the discretized laplacian operator. 
    The values are multiplied by a respective prefactor that is derived from 
    the grid spacing.
    In the end the tensor is reshaped into a 2d matrix that is returned. 
    
    Arguments:
        nx (integer): length of the x dimension
        ny (integer): length of the y dimension
        nz (integer): length of the z dimension
        hx (float): grid size of the x dimension
        hy (float): grid size of the y dimension
        hz (float): grid size of the z dimension
    '''
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
