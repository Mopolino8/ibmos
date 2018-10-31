import functools
import numpy as np
from scipy.special import erf


def solver_cg(A):
    from scipy.sparse.linalg import cg, spilu, LinearOperator

    M = LinearOperator(A.shape, matvec=spilu(A).solve)

    def solver(b, x0=None):
        x, info = cg(A, b, x0=x0, M=M, tol=1e-8)
        if info != 0:
            raise ValueError(f'CG failed: info={info}')
        return x

    return solver,


def solver_minres(A):
    from scipy.sparse.linalg import minres, spilu, LinearOperator

    M = LinearOperator(A.shape, matvec=spilu(A).solve)

    def solver(b, x0=None):
        x, info = minres(A, b, x0=x0, M=M, tol=1e-8)
        if info != 0:
            raise ValueError(f'MINRES failed: info={info}')
        return x

    return solver,


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
    
    #solver.set_iparm(21, 0) #pivoting symmetric indefinite. 1 enable
    #solver.set_iparm(24, 1) #parallel numerial factorization, 1 improved algo
    #solver.set_iparm(25, 1) #parallel backward forward, 1 enabled
    #solver.set_statistical_info_on()

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

    _useUmfpack = spla.dsolve.linsolve.useUmfpack
    spla.use_solver(useUmfpack=True)

    A.indptr = A.indptr.astype(np.int64)
    A.indices = A.indices.astype(np.int64)

    iA=spla.factorized(A)
    def solver(b, x0=None):
        return iA(b)

    spla.use_solver(useUmfpack=_useUmfpack)

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
