import numpy as np
from scipy.special import erf


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
