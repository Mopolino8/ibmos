from dataclasses import *

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from . import quad


@dataclass
class FieldInfo:
    """Stores information of a field defined at the cell centers of a Cartesian grid.

    Attributes
    ----------
    x : np.ndarray
        Location of cell centers in the first direction. Boundaries are not included.
    y : np.ndarray
        Location of cell centers in the second direction. Boundaries are not included.
    dx : np.ndarray
        Cell sizes in the first direction. Not necessarily equal to np.diff(x).
    dy : np.ndarray
        Cell sizes in the second direction. Not necessarily equal to np.diff(y).
    shape : list
        Shape of the field (len(y), len(x))
    size : int
        Total number of grid points.

    """

    x: np.ndarray
    y: np.ndarray

    dx: np.ndarray
    dy: np.ndarray

    shape: list = field(init=False)
    size: int = field(init=False)

    def __post_init__(self):
        """Initialize shape and size."""
        self.shape = len(self.y), len(self.x)
        self.size = self.shape[0] * self.shape[1]

    def weight_height(self):
        """Return height, i.e. dy, weight matrix."""
        return sp.kron(sp.diags(self.dy), sp.eye(self.shape[1]))

    def weight_width(self):
        """Return width, i.e. dx, weight matrix."""
        return sp.kron(sp.eye(self.shape[0]), sp.diags(self.dx))


@dataclass
class Field:
    """Fluid in a rectangular domain.

    Attributes
    ----------
    x : np.ndarray
        x coordinates of the cell vertices.
    y : np.ndarray
        y coordinates of the cell vertices.
    periodic : bool (default, False)
        periodicity in the vertical direction
    xc : np.ndarray
        x coordinates of the cell centers.
    yc : np.ndarray
        y coordinates of the cell centers.
    u : FieldInfo
        u velocity field information.
    v : FieldInfo
        v velocity field information.
    p : FieldInfo
        Pressure field information.

    """

    x: np.ndarray
    y: np.ndarray

    periodic : bool = False

    xc: np.ndarray = field(init=False)
    yc: np.ndarray = field(init=False)

    u: FieldInfo = field(init=False)
    v: FieldInfo = field(init=False)
    p: FieldInfo = field(init=False)

    def __post_init__(self):
        """Initialize u, v and p."""
        self.xc = 0.5 * (self.x[1:] + self.x[:-1])
        self.yc = 0.5 * (self.y[1:] + self.y[:-1])

        self.u = FieldInfo(self.x[1:-1], self.yc, np.diff(self.xc), np.diff(self.y))
        self.p = FieldInfo(self.xc, self.yc, np.diff(self.x), np.diff(self.y))

        # Periodicity in the VERTICAL direction
        if not self.periodic:
            self.v = FieldInfo(self.xc, self.y[1:-1], np.diff(self.x), np.diff(self.yc))
        else:
            yc_vS = (self.yc[0] - self.y[0]) + (self.y[-1] - self.yc[-1])
            self.v = FieldInfo(self.xc, self.y[:-1], np.diff(self.x), np.r_[yc_vS, np.diff(self.yc)])

    def divergence(self):
        Ru = self.p.weight_height()
        Rv = self.p.weight_width()

        DUx, DUxW, DUxE = quad.op(self.x, self.yc, 'x')

        if not self.periodic:
            DVy, DVyS, DVyN = quad.op(self.xc, self.y, 'y', periodic=False)
            return [Ru @ DUx, [Ru @ DUxW, Ru @ DUxE]], [Rv @ DVy, [Rv @ DVyS, Rv @ DVyN]]
        else:
            DVy = quad.op(self.xc, self.y, 'y', periodic=True)
            return [Ru @ DUx, [Ru @ DUxW, Ru @ DUxE]], [Rv @ DVy, []]


    def laplacian(self):
        Mu, Mv = self.u.weight_width(), self.v.weight_height()
        Ru, Rv = self.u.weight_height(), self.v.weight_width()

        DUxx, DUxxW, DUxxE = quad.op(self.x, self.yc, 'xx')

        if not self.periodic:
            yu = np.r_[self.y[0], self.yc, self.y[-1]]
            DUyy, DUyyS, DUyyN = quad.op(self.x[1:-1], yu, 'yy')
            Lu, Lu0 = Ru @ DUxx + Mu @ DUyy, [Ru @ DUxxW, Ru @ DUxxE, Mu @ DUyyS, Mu @ DUyyN]

            DVxx, DVxxW, DVxxE = quad.op(np.r_[self.x[0], self.xc, self.x[-1]], self.y[1:-1], 'xx')
            DVyy, DVyyS, DVyyN = quad.op(self.xc, self.y, 'yy')
            Lv, Lv0 = Mv @ DVxx + Rv @ DVyy, [Mv @ DVxxW, Mv @ DVxxE, Rv @ DVyyS, Rv @ DVyyN]
        else:
            yu = np.r_[self.yc, self.y[-1] + (self.yc[0] - self.y[0])]
            DUyy = quad.op(self.x[1:-1], yu, 'yy', periodic=True)
            Lu, Lu0 = Ru @ DUxx + Mu @ DUyy, [Ru @ DUxxW, Ru @ DUxxE]

            DVxx, DVxxW, DVxxE = quad.op(np.r_[self.x[0], self.xc, self.x[-1]], self.y[:-1], 'xx')
            DVyy = quad.op(self.xc, self.y, 'yy', periodic=True)
            Lv, Lv0 = Mv @ DVxx + Rv @ DVyy, [Mv @ DVxxW, Mv @ DVxxE]

        return [[Lu, Lu0], [Lv, Lv0]]

    def advection(self, u, v, uBC, vBC):
        Mu, Mv = self.u.weight_width(), self.v.weight_height()
        Ru, Rv = self.u.weight_height(), self.v.weight_width()

        dx, dy = np.diff(self.x), np.diff(self.y)

        if not self.periodic:
            uW, uE, uS, uN = uBC
            vW, vE, vS, vN = vBC

            u2 = np.hstack([uW[:, np.newaxis]**2, u**2, uE[:, np.newaxis]**2])

            Nu = dx[1:]*dx[:-1]/(dx[1:] + dx[:-1])*\
                 ((u2[:, 2:] - u2[:, 1:-1])/dx[1:]**2 + (u2[:, 1:-1] - u2[:, :-2])/dx[:-1]**2)

            v2 = np.vstack([vS[np.newaxis, :]**2, v**2, vN[np.newaxis, :]**2])

            Nv = dy[1:, np.newaxis]*dy[:-1, np.newaxis]/(dy[1:] + dy[:-1])[:, np.newaxis]*\
                 ((v2[2:, :] - v2[1:-1, :])/dy[1:, np.newaxis]**2 +
                  (v2[1:-1, :] - v2[:-2, :])/dy[:-1, np.newaxis]**2)

            uv = (u[1:, :]*dy[:-1, np.newaxis] + u[:-1, :]*dy[1:, np.newaxis])/(dy[1:] + dy[:-1])[:, np.newaxis]*\
                 (v[:, 1:]*dx[:-1] + v[:, :-1]*dx[1:])/(dx[1:] + dx[:-1])

            uvS = uS*(vS[1:]*dx[:-1] + vS[:-1]*dx[1:])/(dx[:-1] + dx[1:])
            uvN = uN*(vN[1:]*dx[:-1] + vN[:-1]*dx[1:])/(dx[:-1] + dx[1:])

            uvW = vW*(uW[1:]*dy[:-1] + uW[:-1]*dy[1:])/(dy[:-1] + dy[1:])
            uvE = vE*(uE[1:]*dy[:-1] + uE[:-1]*dy[1:])/(dy[:-1] + dy[1:])

            Nu += np.diff(np.vstack([uvS, uv, uvN]), axis=0)/dy[:, np.newaxis]
            Nv += np.diff(np.hstack([uvW[:, np.newaxis], uv, uvE[:, np.newaxis]]), axis=1)/dx
        else:
            uW, uE = uBC
            vW, vE = vBC

            u2 = np.hstack([uW[:, np.newaxis] ** 2, u ** 2, uE[:, np.newaxis] ** 2])

            Nu = dx[1:]*dx[:-1]/(dx[1:] + dx[:-1])*\
                 ((u2[:, 2:] - u2[:, 1:-1])/dx[1:]**2 + (u2[:, 1:-1] - u2[:, :-2])/dx[:-1]**2)

            v2 = np.vstack([v[-1, :]**2, v**2, v[0, :]**2])

            dyv = np.r_[dy[-1], dy]

            Nv = dyv[1:, np.newaxis]*dyv[:-1, np.newaxis]/(dyv[1:] + dyv[:-1])[:, np.newaxis]*\
                ((v2[2:, :] - v2[1:-1, :])/dyv[1:, np.newaxis]**2 +
                 (v2[1:-1, :] - v2[:-2, :])/dyv[:-1, np.newaxis]**2)

            uv = (u[:, :]*np.roll(dy, 1)[:, np.newaxis] + np.roll(u[:, :], 1, axis=0)*dy[:, np.newaxis]) / (np.roll(dy, 1) + dy)[:, np.newaxis] * \
                 (v[:, 1:]*dx[:-1] + v[:, :-1]*dx[1:])/(dx[1:] + dx[:-1])

            uvW = vW*(uW[:]*dyv[:-1] + np.r_[uW[-1], uW[:-1]]*dyv[1:])/(dyv[:-1] + dyv[1:])
            uvE = vE*(uE[:]*dyv[:-1] + np.r_[uE[-1], uE[:-1]]*dyv[1:])/(dyv[:-1] + dyv[1:])

            Nu += np.diff(np.vstack([uv, uv[0, :]]), axis=0)/dy[:, np.newaxis]
            Nv += np.diff(np.hstack([uvW[:, np.newaxis], uv, uvE[:, np.newaxis]]), axis=1)/dx

        return Mu@Ru@Nu.ravel(), Mv@Rv@Nv.ravel()

    def linearized_advection(self, u0, v0, u0BC, v0BC, test=True):
        n, m = self.p.shape
        h = 1e-8

        # Tiling width
        sw = 3

        if self.periodic and (n%3)!=0:
            raise ValueError ("In the periodic case, n must be divisible by 3: n=%d, n/3 = %f"%(n, n/3))

        # aux vectors for building the block matrices that form N
        # N = [Nuu, Nuv]
        #     [Nvu, Nvv]
        data_Nuu, row_Nuu, col_Nuu = np.asarray([]), np.asarray([]), np.asarray([])
        data_Nvu, row_Nvu, col_Nvu = np.asarray([]), np.asarray([]), np.asarray([])
        data_Nuv, row_Nuv, col_Nuv = np.asarray([]), np.asarray([]), np.asarray([])
        data_Nvv, row_Nvv, col_Nvv = np.asarray([]), np.asarray([]), np.asarray([])

        # stencils
        suu = [[0, -1], [0, 0], [0, 1], [-1, 0], [1, 0]]
        svv = [[0, -1], [0, 0], [0, 1], [-1, 0], [1, 0]]

        if not self.periodic:
            svu = [[-1, 0], [-1, 1], [0, 0], [0, 1]]
            suv = [[1, -1], [0, -1], [0, 0], [1, 0]]
        else:
            svu = [[1, 0], [1, 1], [0, 0], [0, 1]]
            suv = [[-1, -1], [0, -1], [0, 0], [-1, 0]]

        # obtain Nuu and Nvu
        for idxj in range(sw):
            for idxi in range(sw):
                u = np.zeros_like(u0)
                u[idxj::sw, idxi::sw] = 1

                idx = np.where(u)
                tmpidx = np.arange(u0.size).reshape(u0.shape)[idx]

                uidx = -np.ones(u0.shape, dtype=int)
                for suuk in suu:
                    jj, ii = idx[0] + suuk[0], idx[1] + suuk[1]
                    if self.periodic:
                        jj[jj == -1] = u0.shape[0] - 1
                        jj[jj == u0.shape[0]] = 0

                    mask = (0 <= jj) * (jj < u0.shape[0]) * (0 <= ii) * (ii < u0.shape[1])
                    uidx[jj[mask], ii[mask]] = tmpidx[mask]

                vidx = -np.ones(v0.shape, dtype=int)
                for svuk in svu:
                    jj, ii = idx[0] + svuk[0], idx[1] + svuk[1]
                    if self.periodic:
                        jj[jj == -1] = u0.shape[0] - 1
                        jj[jj == u0.shape[0]] = 0

                    mask = (0 <= jj) * (jj < v0.shape[0]) * (0 <= ii) * (ii < v0.shape[1])
                    vidx[jj[mask], ii[mask]] = tmpidx[mask]

                Nu1, Nv1 = self.advection(u0 + 1j * h * u, np.asarray(v0, dtype=complex), u0BC, v0BC)
                Nu, Nv = Nu1.imag / h, Nv1.imag / h

                mask = uidx.ravel() >= 0
                row_Nuu = np.concatenate([row_Nuu, np.arange(u0.size)[mask]])
                col_Nuu = np.concatenate([col_Nuu, uidx.ravel()[mask]])
                data_Nuu = np.concatenate([data_Nuu, Nu[mask]])

                mask = vidx.ravel() >= 0
                row_Nvu = np.concatenate([row_Nvu, np.arange(v0.size)[mask]])
                col_Nvu = np.concatenate([col_Nvu, vidx.ravel()[mask]])
                data_Nvu = np.concatenate([data_Nvu, Nv[mask]])

        # obtain Nuv and Nvv
        for idxj in range(sw):
            for idxi in range(sw):
                v = np.zeros_like(v0)
                v[idxj::sw, idxi::sw] = 1

                idx = np.where(v)
                tmpidx = np.arange(v0.size).reshape(v0.shape)[idx]

                vidx = -np.ones(v0.shape, dtype=int)
                for svvk in svv:
                    jj, ii = idx[0] + svvk[0], idx[1] + svvk[1]
                    if self.periodic:
                        jj[jj == -1] = u0.shape[0] - 1
                        jj[jj == u0.shape[0]] = 0

                    mask = (0 <= jj) * (jj < v0.shape[0]) * (0 <= ii) * (ii < v0.shape[1])
                    vidx[jj[mask], ii[mask]] = tmpidx[mask]

                uidx = -np.ones(u0.shape, dtype=int)
                for suvk in suv:
                    jj, ii = idx[0] + suvk[0], idx[1] + suvk[1]
                    if self.periodic:
                        jj[jj == -1] = u0.shape[0] - 1
                        jj[jj == u0.shape[0]] = 0

                    mask = (0 <= jj) * (jj < u0.shape[0]) * (0 <= ii) * (ii < u0.shape[1])
                    uidx[jj[mask], ii[mask]] = tmpidx[mask]

                Nu1, Nv1 = self.advection(np.asarray(u0, dtype=complex), v0 + 1j * h * v, u0BC, v0BC)
                Nu, Nv = Nu1.imag / h, Nv1.imag / h

                mask = uidx.ravel() >= 0
                row_Nuv = np.concatenate([row_Nuv, np.arange(u0.size)[mask]])
                col_Nuv = np.concatenate([col_Nuv, uidx.ravel()[mask]])
                data_Nuv = np.concatenate([data_Nuv, Nu[mask]])

                mask = vidx.ravel() >= 0
                row_Nvv = np.concatenate([row_Nvv, np.arange(v0.size)[mask]])
                col_Nvv = np.concatenate([col_Nvv, vidx.ravel()[mask]])
                data_Nvv = np.concatenate([data_Nvv, Nv[mask]])

        Nuu = sp.coo_matrix((data_Nuu, (row_Nuu, col_Nuu)), shape=(u0.size, u0.size))
        Nvu = sp.coo_matrix((data_Nvu, (row_Nvu, col_Nvu)), shape=(v0.size, u0.size))
        Nuv = sp.coo_matrix((data_Nuv, (row_Nuv, col_Nuv)), shape=(u0.size, v0.size))
        Nvv = sp.coo_matrix((data_Nvv, (row_Nvv, col_Nvv)), shape=(v0.size, v0.size))

        N = sp.bmat([[Nuu, Nuv], [Nvu, Nvv]]).tocsr()
        N.eliminate_zeros()

        if test:
            # First random u
            u = np.random.random(u0.shape)
            v = np.zeros_like(v0)

            Nu1, Nv1 = self.advection(u0 + 1j * h * u, np.asarray(v0, dtype=complex), u0BC, v0BC)
            NU1 = np.concatenate([Nu1.imag / h, Nv1.imag / h])
            NU2 = N @ np.r_[u.ravel(), v.ravel()]
            NUerr = la.norm(NU2 - NU1) / la.norm(NU1)

            # Then random v
            u = np.zeros(u0.shape)
            v = np.random.random(v0.shape)

            α = la.norm(v, np.inf)
            Nu1, Nv1 = self.advection(np.asarray(u0, dtype=complex), v0 + 1j * h * v, u0BC, v0BC)
            NU1 = np.concatenate([Nu1.imag / h, Nv1.imag / h])
            NU2 = N @ np.r_[u.ravel(), v.ravel()]
            NVerr = la.norm(NU2 - NU1) / la.norm(NU1)

            if NUerr > 1e-13 or NVerr > 1e-13:
                raise ValueError("Linearization check failed: Nuerr = %f and Nverr = %f" % (NUerr, NVerr))

        return N
