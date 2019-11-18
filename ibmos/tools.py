import functools
import numpy as np
from scipy.special import erf
import scipy.sparse as sp


def solver_pardiso(A):
    """ 
    Return a function for solving a sparse linear system using PARDISO.
    
    Parameters
    ----------
    A : (N, N) array_like
        Input.
        
    Returns
    -------
    solve : callable
        To solve the linear system of equations given in `A`, the `solve`
        callable should be passed an ndarray of shape (N,).
        
    """
    
    from pypardiso import spsolve, PyPardisoSolver
    
    pypardisosolver = PyPardisoSolver(mtype=11)
    pypardisosolver.set_iparm(1, 1)
    pypardisosolver.set_iparm(2, 2)
    pypardisosolver.set_iparm(10, 10) # pivot drop tol. (was 13)
    pypardisosolver.set_iparm(11, 1) # 0 disable, 1 enable scaling vectors.
    pypardisosolver.set_iparm(13, 0) # 0 disable, 1 matching normal, 2 advanced matching
    
    #pypardisosolver.set_iparm(21, 0) #pivoting symmetric indefinite. 1 enable
    #pypardisosolver.set_iparm(24, 1) #parallel numerial factorization, 1 improved algo
    #pypardisosolver.set_iparm(25, 1) #parallel backward forward, 1 enabled
    #pypardisosolver.set_statistical_info_on()

    def solver(b, x0=None):
        return spsolve(A, b, squeeze=False, solver=pypardisosolver)

    return solver, pypardisosolver


def solver_superlu(A):
    """ 
    Return a function for solving a sparse linear system using SuperLU.
    
    Parameters
    ----------
    A : (N, N) array_like
        Input.
        
    Returns
    -------
    solve : callable
        To solve the linear system of equations given in `A`, the `solve`
        callable should be passed an ndarray of shape (N,).
        
    """
    
    import scipy.sparse.linalg as spla

    _useUmfpack = spla.dsolve.linsolve.useUmfpack
    spla.use_solver(useUmfpack=False)

    iA = spla.factorized(A)

    def solver(b, x0=None):
        return iA(b)

    spla.use_solver(useUmfpack=_useUmfpack)

    return solver,


def solver_umfpack(A):
    """ 
    Return a function for solving a sparse linear system using UMFPACK.
    
    Parameters
    ----------
    A : (N, N) array_like
        Input.
        
    Returns
    -------
    solve : callable
        To solve the linear system of equations given in `A`, the `solve`
        callable should be passed an ndarray of shape (N,).
        
    """
    
    import scipy.sparse.linalg as spla

    #_useUmfpack = spla.dsolve.linsolve.useUmfpack
    spla.use_solver(useUmfpack=True)

    A.indptr = A.indptr.astype(np.int64)
    A.indices = A.indices.astype(np.int64)

    iA=spla.factorized(A)
    def solver(b, x0=None):
        return iA(b)

    #spla.use_solver(useUmfpack=_useUmfpack)

    return solver,


def solver_pcg_ilu(A):
    """ 
    Return a function for solving a sparse linear system using PCG with ILU.
    
    Parameters
    ----------
    A : (N, N) array_like
        Input.
        
    Returns
    -------
    solve : callable
        To solve the linear system of equations given in `A`, the `solve`
        callable should be passed an ndarray of shape (N,).
        
    """

    from scipy.sparse.linalg import cg, spilu, LinearOperator

    M = LinearOperator(A.shape, matvec=spilu(A).solve)

    def solver(b, x0=None):
        x, info = cg(A.T, b, x0=x0, M=M)
        if info != 0:
            raise ValueError(f'CG failed: info={info}')
        return x

    return solver,


def solver_pcg_amg(A):
    """ 
    Return a function for solving a sparse linear system using PCG with AMG.
    
    Parameters
    ----------
    A : (N, N) array_like
        Input.
        
    Returns
    -------
    solve : callable
        To solve the linear system of equations given in `A`, the `solve`
        callable should be passed an ndarray of shape (N,).
        
    """

    from scipy.sparse.linalg import cg
    from pyamg import rootnode_solver
    
    M = rootnode_solver(A.T).aspreconditioner(cycle='V')

    def solver(b, x0=None):
        x, info = cg(A.T, b, x0=x0, M=M)
        if info != 0:
            raise ValueError(f'CG failed: info={info}')
        return x

    return solver,


def solver_pminres_ilu(A):
    """ 
    Return a function for solving a sparse linear system using MINRES with ILU.
    
    Parameters
    ----------
    A : (N, N) array_like
        Input.
        
    Returns
    -------
    solve : callable
        To solve the linear system of equations given in `A`, the `solve`
        callable should be passed an ndarray of shape (N,).
        
    """

    from scipy.sparse.linalg import minres, spilu, LinearOperator

    M = LinearOperator(A.shape, matvec=spilu(A.solve))

    def solver(b, x0=None):
        x, info = minres(A.T, b, x0=x0, M=M)
        if info != 0:
            raise ValueError(f'MINRES failed: info={info}')
        return x

    return solver,


def solver_default():
    """ 
    Return (fastest?) available sparse direct solver.
    
    Returns
    -------
    solver : callable
        To create a solver, call `solver` with matrix `A` (N x N) as the first parameter.
        
    """
    
    try:
        import pypardiso
        return solver_pardiso
    except ImportError:
        return solver_superlu


def stretching(n, dn0, dn1, ns, ws=12, we=12, maxs=0.04):
    """Return stretched segment.

    Parameters
    ----------
    n : int
        Total number of points.
    dn0 : float
        Initial grid spacing.
    dn1 : float
        Final grid spacing.
    ns : int
        Number of grid points with spacing equal to dn0
    ws : int, optional
        Number of grid points from stretching zero to stretching maxs
    we : int, optional
        Number of grid points from stretching maxs to stretching zero
    maxs : float, optional
        Maximum stretching (ds_i+1 - ds_i)/ds_i

    Returns
    -------
    f: np.ndarray
        One-dimensional np.array

    """

    ne = ns + np.log(dn1 / dn0) / np.log(1 + maxs)

    s = np.array([maxs * 0.25 * (1 + erf(6 * (x - ns) / (ws))) * (1 - erf(6 * (x - ne) / we)) for x in range(n)])

    f_ = np.empty(s.shape)
    f_[0] = dn0
    for k in range(1, len(f_)):
        f_[k] = f_[k - 1] * (1 + s[k])
    f = np.empty(s.shape)
    f[0] = 0.0
    for k in range(1, len(f)):
        f[k] = f[k - 1] + f_[k]

    return f

def perm_from_colors(colors):
    """ Return permutation indices from colors (ascending order).

    Parameters
    ----------
    colors : np.ndarray
        Colors.

    Returns
    -------
    perm: np.ndarray
        Permutation.

    """ 

    idx = np.arange(len(colors))
    perm = np.concatenate([idx[color == colors] for color in np.unique(colors)])

    return perm

def mat_from_perm(perm):
    """ Return permutation matrix from permutation indices.

    Parameters
    ----------
    perm : np.ndarray
        Permutation indices.

    Returns
    -------
    P: np.ndarray
        Permutation matrix.

    """ 

    return sp.eye(len(perm), dtype=int).tocsr()[perm, :]

def iperm(perm):
    """ Return inverse permutation indices.

    Parameters
    ----------
    perm : np.ndarray
        Permutation indices.

    Returns
    -------
    iperm: np.ndarray
        Inverse permutation indices.

    """ 
    iperm = np.empty_like(perm)
    iperm[perm] = np.arange(perm.size)

    return iperm

def submat_from_colors(M, ci, cj, colorsi, colorsj=None):
    """ Return matrix from colored rows and columns.

    Parameters
    ----------
    M : matrix
        Matrix operator.
    ci : int
        Row color.
    cj : int
        Column color.
    colorsi : np.ndarray
        Row coloring.
    colorsj : np.ndarray, optional
        Column color (defaults to colorsi).

    Returns
    -------
    Mij: matrix
        Submatrix Mij.

    """ 
    if colorsj is None:
        colorsj = colorsi
        
    return M[colorsi==ci, :][:, colorsj==cj].tocsr()

def submats_from_colors(M, colorsi, colorsj=None):
    """ Divide matrix into submatrices according to coloring.

    Parameters
    ----------
    M : matrix
        Matrix.
    colorsi : np.ndarray
        Row coloring.
    colorsj : np.ndarray, optional
        Column color (defaults to colorsi).

    Returns
    -------
    Mc: list
        List of list of matrices.

    """ 

    if colorsj is None:
        colorsj = colorsi

    ucolorsi = np.unique(colorsi)
    ucolorsj = np.unique(colorsj)

    Mc =[ [submat_from_colors(M, ci, cj, colorsi, colorsj) for cj in ucolorsj] for ci in ucolorsi]

    return Mc

def subvec_from_colors(v, ci, colors):
    """ Return vector from color.

    Parameters
    ----------
    v : np.ndarray
        Vector.
    ci : int
        Row color.
    colors : np.ndarray
        Row coloring.

    Returns
    -------
    vc: np.ndarray
        Vector.

    """ 
    return v[colors==ci]

def subvecs_from_colors(v, colors):
    """ Divide vector into subvectors according to coloring.

    Parameters
    ----------
    v : np.ndarray
        Vector.
    colors : np.ndarray
        Row coloring.

    Returns
    -------
    vc: list
        List of vectors.

    """ 

    unique_colors = np.unique(colors)
    vc = [subvec_from_colors(v, colors, ci) for ci in unique_colors]

    return vc

def dirichlet_kernel_equispaced(x, n): 
    """ Dirichlet kernel for equispaced data.

    Parameters
    ----------
    x : float
        Point
    n : number of grid points
        Initial grid spacing.

    Returns
    -------
    D: float
        Value of the Dirichlet kernel at x

    """ 
    D = np.sinc(N*x/2)/np.sinc(x/2)*(1 if N%2 else np.cos(np.pi*x/2))
    return D

def interpolation_matrix(xi, xj, kernel=dirichlet_kernel_equispaced):
    """Return interpolation matrix from xj to xi.

    Parameters
    ----------
    xi : np.ndarray
        Destination
    xj : np.ndarray
        Origin.
    kernel : function, optional
        Interpolation kernel f(x, n)

    Returns
    -------
    S: np.ndarray
        interpolation matrix

    """

    n = len(xj)
    scale = (xj[1]-xj[0])*n/2
    S = kernel((xi[:, np.newaxis]-xj[np.newaxis, :])/scale, n)
    return 
