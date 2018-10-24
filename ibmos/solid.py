from dataclasses import *

import numpy as np
import scipy.sparse as sp


def _interp(delta, nelem, ξ, x, dx, normalized=None):
    """Auxiliary function for 1D interpolation."""
    ξ_x = np.argmin(np.abs(ξ[:, np.newaxis] - x[np.newaxis, :]), axis=1)
    ξ_dx = dx[ξ_x]

    scale_factor = ξ_dx if normalized else np.ones_like(ξ_dx)

    e_i, e_j, e_val = [], [], []
    for j in range(len(ξ)):
        xj = ξ_x[j]

        e_i.append(j * np.ones(2 * nelem + 1))
        e_j.append(np.arange(xj - nelem, xj + nelem + 1))
        deltaj = delta((ξ[j] - x[xj - nelem:xj + nelem + 1]), ξ_dx[j])
        e_val.append(scale_factor[j] * deltaj)

    e_i, e_j, e_val = np.asarray(e_i).ravel(), np.asarray(e_j).ravel(), np.asarray(e_val).ravel()
    E_ = sp.coo_matrix((e_val, (e_i, e_j)), shape=(len(ξ), len(x)))
    return E_


@dataclass
class Solid:
    name: str
    ξ: np.ndarray
    η: np.ndarray
    ds: np.ndarray

    δ: None
    n: int

    l: int = field(init=False)

    def __post_init__(self):
        self.l = len(self.ξ)

    def interpolation(self, field, normalized=True):
        Ey = sp.kron(_interp(self.δ, self.n, self.η, field.y, field.dy, normalized), np.ones_like(field.x)).tocsr()
        Ex = sp.kron(np.ones_like(field.y), _interp(self.δ, self.n, self.ξ, field.x, field.dx, normalized)).tocsr()
        E = Ey.multiply(Ex)
        return E

    def regularization(self, field):
        return self.interpolation(field, False).T @ sp.diags(self.ds)
