from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from .flow import Field
from .tools import solver_default, interpolation_matrix
from .solver import Solver
from .domaindecomposition import PoincareSteklov, Matrix12

class SolverSlidingPlane(Solver):
    """Flow solver based on the Projection-based Immersed Boundary Method 
       with a sliding plane at xSP."""

    xSP: float

    def __init__(self, x, y, xSP, *solids, iRe=1.0, Co=0.5, advection=True, 
                 solver=solver_default()):
        """Initialize solver.

        Parameters
        ----------
        x : np.array
            Coordinates of the vertices (x direction).
        y : np.array
            Coordinates of the vertices (y direction).
        xSP: float
            x coordinate of the sliding plane.
        solids : list, optional
            List of solids. 
        iRe : float, optional
            Inverse of the Reynolds number.
        Co : float, optional
            Courant number.
        advection : bool, optional
            Enable or disable advection terms
        solver : callable, optional
            `solver(A)[0]` that returns linear solver. """

        super().__init__(x, y, *solids, iRe=iRe, Co=Co, periodic=True, advection=advection, 
                        fractionalStep=True, pZero=None, solver=solver)

        self.set_sliding_plane(xSP)


    def set_sliding_plane(self, xSP):
        """ Update coordinate of the sliding plane

        Parameters
        ----------
        xSP : float
            Coordinate of the sliding plane
        
        Note
        ----
        The sliding plane must not intersect an immersed boundary and
        must be located at least two to three points away from the surfaces."""
    
        self.xSP = xSP

        self.iuSP = np.nonzero(self.fluid.u.x > self.xSP)[0][0]
        self.ivSP = np.nonzero(self.fluid.v.x > self.xSP)[0][0]
        self.ipSP = np.nonzero(self.fluid.p.x > self.xSP)[0][0]

        self.stepsInitialized = False

        self.set_pZero(None)
        c12_0 = self.colors_12_from_0()[1]

        QBNQT = PoincareSteklov(self.propagator(True)[0][1], c12_0[self.pStart:])
        
        # find first point at the sliding plane to pin pressure
        self.set_pZero(np.nonzero(QBNQT.colors_12[:self.pEnd-self.pStart]==1)[0][0]-1)


    def colors_12_from_0(self):
        """ Return coloring.

        Note
        ----
        0 is left domain and 1 is right domain.
        """

        xp0, x = self.state_vector_x()

        return 1*(xp0 > self.xSP), 1*(x > self.xSP)


    def steps(self, x, fuBC, fvBC, fsBC=(), t0=0.0, ySP0 = 0.0, vSP=0.0, outflowEast=False,
              number=1, saveEvery=None, verbose=1, checkSolvers=False):
        """Time-step the governing equations.

        Parameters
        ----------
        x : np.ndarray
            Initial condition (packed state-vector).
        fuBC : list
            List of functions of (x, t) and (y, t) for the West and East
            boundary conditions for the horizontal component of the velocity field.
        fvBC : list
            List of functions of (x, t) and (y, t) for the West, East
            boundary conditions for the vertical component of the velocity field.
        fsBC : list, optional
            List of functions of (ξ, η, t) for the horizontal and vertical 
            component of the velocity on the immersed boundaries.
        t0 : float, optional
            Initial time.
        ySP0 : float, optional
            vertical displacement of domain 2 with respect to domain 1.
        vSP : float, optional
            vertical speed of domain 2 with respect to domain 1.
        outflowEast : bool, optional
            East boundary has outflow boundary condition. Note that uBC[1]
            and vBC[1] are updated every each iteration.
        number : int, optional
            Number of time steps.
        saveEvery : int, optional
            Specify how often flow fields are stored. By default, only the
            last one is returned.
        verbose : int, optional
            Specify how often x_2, dx/dt_2 and forces are displayed.
        checkSolvers : bool, optional
            Check linear solvers.

        Returns
        -------
        xres : (np.ndarray)
            Flow fields sampled every saveEvery steps.
        tres: list (np.ndarray)
            Time.
        rres: list (np.ndarray)
            Numer of rotations since startup.
        ySPres: list (np.ndarray)
            Vertical displacement.
        infodict: dict
            Norm of the state vector, temporal derivative, and forces.
        """

        if not self.stepsInitialized:
            if verbose:
                print("Initializing solver...", end="")

            self.A, self.B = self.propagator(self.fractionalStep)

            # Split colors, slices and names into velocity fields plus the rest
            cuv, cpfs = np.split(self.colors_12_from_0()[1], (self.pStart,))
            juv, jpfs = np.split(self.state_vector_j()[1], (self.pStart,))
            nuv, npfs = np.split(self.state_vector_name()[1], (self.pStart,))

            self.A[0] = PoincareSteklov(self.A[0], cuv, juv, nuv)
            self.A[1] = PoincareSteklov(self.A[1], cpfs, jpfs, npfs)

            self.B[0] = Matrix12(self.B[0],  cuv, cuv, juv, nuv)
            self.B[1] = Matrix12(self.B[1],  cuv, cuv, juv, nuv)
            self.B[2] = Matrix12(self.B[2], cpfs, cuv, juv, nuv)

            self.A[0].factorize(solver=self.solver)
            self.A[1].factorize(solver=self.solver)

            self.interpolation_matrices_update(ySP0, *self.A, *self.B)

            if verbose:
                print("done.")

            self.stepsInitialized = True


        if saveEvery is None:
            saveEvery = number

        xres, tres, rSPres, ySPres = [], [], [], []

        uBC, vBC = self.eval_uvBC(t0, fuBC, fvBC)

        # Contribution of the laplacian at the sliding plane due to 
        # the relative velocity between the two subdomains
        # Man, this one wasn't obvious !!

        bvSP = np.zeros(self.pStart)
        if vSP != 0:
            from .tools import submat_from_colors

            c = self.colors_12_from_0()[1][:self.pStart]
            n = self.state_vector_name()[1][:self.pStart]

            Lv12 = submat_from_colors(self.laplacian[1][0], 0, 1, c[n=='v'])
            Lv21 = submat_from_colors(self.laplacian[1][0], 1, 0, c[n=='v'])

            bvSP[(c==0)*(n=='v')] += vSP*self.iRe*Lv12@((self.B[0].S2@((n=='v')[c==1]))[n[c==1]=='v'])
            bvSP[(c==1)*(n=='v')] -= vSP*self.iRe*Lv21@((self.B[0].S1@((n=='v')[c==0]))[n[c==0]=='v'])
    
        # Advection terms at the CURRENT time step.
        if self.advection:
            Nu, Nv = self.advection_(*super().reshape(super().unpack(x), p0=0)[:2], uBC, vBC, ySP0, vSP)
        else:
            Nu, Nv = np.zeros(self.fluid.u.size), np.zeros(self.fluid.v.size) 

        Num1, Nvm1 = Nu, Nv

        # Dictionary with output variables
        header = ['t', 'rSP', 'ySP', 'x_2', 'dxdt_2']
        header.extend(chain(*[(f'{solid.name}_fx', f'{solid.name}_fy')
                              for solid in self.solids]))
        if outflowEast:
            header.append('Uinf@outlet')
        if checkSolvers:
            header.extend(['rel.error(A)', 'rel.error(C)'])

        # Create dictionary
        infodict = dict(zip(header, (np.empty(number) for _ in header)))

        # If verbose, print header.
        if verbose:
            print("       k", "".join((f'{elem:>12} ' for elem in header)))

        # Main loop.
        try:
            for k in range(number):
                t = t0 + (k+1)*self.dt
                rSP = int(((t-t0)*vSP)//(self.fluid.y[-1] - self.fluid.y[0]))
                ySP = (ySP0 + (t-t0)*vSP)%(self.fluid.y[-1] - self.fluid.y[0])

                self.interpolation_matrices_update(ySP, *self.A, *self.B)

                infodict['t'][k] = t
                infodict['rSP'][k] = rSP
                infodict['ySP'][k] = ySP

                uBCp1, vBCp1 = self.eval_uvBC(t, fuBC, fvBC)
                sBCp1 = self.eval_sBC(t, fsBC)

                # if east boundary is an outflow, use convective outflow
                if outflowEast:
                    uBCp1[1], vBCp1[1] = uBC[1], vBC[1]

                # Contribution of the boundary conditions to the right-hand-side.
                bc = self.boundary_condition_terms(uBC, vBC, uBCp1, vBCp1, *sBCp1)

                # Interpolate x, N and Nm1 into the new grid
                # NOTE: THERE IS NO NEED TO INTERPOLATE P BECAUSE IT IS NOT USED!!
                unew, vnew = self.sliding_plane_interpolation(*self.reshape(self.unpack(x), p0=0)[:2], ySP=-self.dt*vSP)
                x[:self.pStart] = np.r_[unew.ravel(), vnew.ravel()]

                if self.advection:
                    Nu,   Nv   = self.sliding_plane_interpolation(Nu  , Nv  , ySP=-self.dt*vSP)
                    Num1, Nvm1 = self.sliding_plane_interpolation(Num1, Nvm1, ySP=-self.dt*vSP)

                # Compute next time step. Time consuming part
                b = self.B[0].dot(x[:self.pStart]) + bc[:self.pStart] + bvSP
                b += -1.5 * np.r_[Nu.ravel(), Nv.ravel()] + 0.5 * np.r_[Num1.ravel(), Nvm1.ravel()]

                qast = self.A[0].solve(b, q0_ = None if k==0 else qast)
                λ = self.A[1].solve(self.B[2].dot(qast) - bc[self.pStart:], q0_ = None if k==0 else λ)

                xp1 = np.r_[qast - self.B[1].dot(self.B[2].dotT(λ)), λ]

                if checkSolvers:
                    errA = la.norm(self.A[0].dot(qast) - b)/la.norm(b)
                    errC = la.norm(self.A[1].dot(λ) - self.B[2].dot(qast) + bc[self.pStart:])/ \
                           la.norm(self.B[2].dot(qast) - bc[self.pStart:])
                    infodict['rel.error(A)'][k] = errA
                    infodict['rel.error(C)'][k] = errC

                infodict['x_2'][k] = la.norm(xp1)
                infodict['dxdt_2'][k] = la.norm(xp1-x)/self.dt

                if self.solids:
                    fp1 = self.unpack(xp1)[3:]
                    for l, solid in enumerate(self.solids):
                        infodict[f'{solid.name}_fx'][k] = 2*np.sum(fp1[l][0])
                        infodict[f'{solid.name}_fy'][k] = 2*np.sum(fp1[l][1])

                if outflowEast and self.advection:
                    u, v = self.reshape(self.unpack(xp1), p0=0)[:2]

                    Uinf = np.sum(self.fluid.u.dy*u[:,-1])/(self.fluid.y[-1]-self.fluid.y[0])
                    infodict['Uinf@outlet'][k] = Uinf
                    dx = (self.fluid.x[-1]-self.fluid.x[-2])
                    uBCp1[1][:] = uBC[1][:] - Uinf*self.dt/dx*(uBC[1][:] - u[:,-1])
                    vBCp1[1][:] = vBC[1][:] - Uinf*self.dt/dx*(vBC[1][:] - v[:,-1])

                # If reportEvery is not None, print current step, time, residuals and,
                # if we have immersed boundaries, print also the forces.
                if verbose and ((k + 1) % verbose == 0 or (k + 1) == number):
                    print(f"{k+1:8}", "".join((f'{infodict[elem][k]: 12.5e} ' for elem in header)))

                # Prepare for the next time step
                x = xp1
                Num1, Nvm1 = Nu, Nv
                uBC, vBC = uBCp1, vBCp1

                if k != number - 1 and self.advection:
                    Nu, Nv = self.advection_(*self.reshape(self.unpack(x), p0=0)[:2], uBC, vBC, ySP, vSP)

                # Append vector to xres?
                if (k + 1) % saveEvery == 0:
                    xres.append(x.copy())
                    tres.append(t)
                    rSPres.append(rSP)
                    ySPres.append(ySP)

        except KeyboardInterrupt:
            print("KeyboardInterrupt at t =", t0 + k*self.dt)
            xres.append(x)
            tres.append(t0 + k*self.dt)
            rSPres.append( (k*self.dt*vSP)//(self.fluid.y[-1] - self.fluid.y[0]) )
            ySPres.append( (ySP0 + k*self.dt*vSP)%(self.fluid.y[-1] - self.fluid.y[0]) )
        except ValueError:
            print("ValueError at t =", t0 + k*self.dt)
            xres.append(x)
            tres.append(t0 + k*self.dt)
            rSPres.append( (k*self.dt*vSP)//(self.fluid.y[-1] - self.fluid.y[0]) )
            ySPres.append( (ySP0 + k*self.dt*vSP)%(self.fluid.y[-1] - self.fluid.y[0]) )

        # Return state vectors
        return np.squeeze(xres), np.squeeze(tres), np.squeeze(rSPres), np.squeeze(ySPres), infodict


    def plot_domain_12(self, ySP=0, equal=True, borders=False, 
                       figsize=(6, 6), xlim=(), ylim=(), ms=2):
        """Plot domain and immersed boundaries.
        
        Parameters
        ----------
        ySP: float, optional
            Vertical displacement of the second subdomain.
        equal: bool, optional
            Use equal axes.
        figsize: optional, tuple
            Size of the figure.
        xlim: optional, tuple
            x limits.
        ylim: optional, tuple
            y limits.
        ms: optional, float
            marker size.
        """

        fig = plt.figure(figsize=figsize)
        plt.title('Fluid domain and immersed boundaries')

        x_pZero, x = self.state_vector_x()
        y_pZero, y = self.state_vector_y()

        for ck, d, c in zip(np.unique(self.colors_12_from_0()[1]), (0, ySP), ('b', 'r')):
            x_ = self.unpack_color(x, self.colors_12_from_0()[1], ck)
            y_ = self.unpack_color(y, self.colors_12_from_0()[1], ck)

            plt.plot(x_[0], y_[0]+d, c+'>', ms=ms)
            plt.plot(x_[1], y_[1]+d, c+'^', ms=ms)
            plt.plot(x_[2], y_[2]+d, c+'s', ms=ms)

            for xk, yk in zip(x_[3:], y_[3:]):
                plt.plot(xk[0], yk[0]+d, c+'-*', lw=0.5, ms=ms)

        plt.plot(x_pZero, y_pZero, 'ok', ms=4*ms)


        if equal:
            plt.axis('equal')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        if xlim:
            plt.xlim(*xlim)

        if ylim:
            plt.ylim(*ylim)

        if borders:
            plt.vlines(self.xSP, ymin=np.min(self.fluid.y), ymax=np.max(self.fluid.y),
                       colors='k', lw=2, linestyles='dashdot', zorder=10, label='S.P.')


    def plot_field_12(self, q, ySP=0, vSP=0, colorbar=False, equal=True, repeat=True, 
                     stitch=True, borders=False, figsize=(8, 3), xlim=(), ylim=()):
        """Plot field.

        Parameters
        ----------
        q: np.ndarray
            State vector subdomain 2(packed).
        ySP: float, optional
            Vertical displacement of the second subdomain.
        vSP: float, optional
            Vertical speed of the second subdomain.
        colorbar: bool, optional
            Display colorbar.
        equal: bool, optional
            Use equal axes.
        repeat: bool, optional
            Repeat if the flow field is periodic.
        stitch: bool, optional
            Fill gaps between the subdomains.
        borders: bool, optional
            Draw borders and sliding plane.
        figsize: optional, tuple
            Size of the figure.
        xlim: optional, tuple
            x limits.
        ylim: optional, tuple
            y limits.

        """

        fig = plt.figure(figsize=figsize)

        Ly = self.fluid.y[-1] - self.fluid.y[0]

        plots = ('u', 'v', 'p')

        for k, name in enumerate(plots):
            plt.subplot(1, len(plots), k + 1)
            plt.title(name)

            vmin, vmax = np.min(self.unpack(q)[k]), np.max(self.unpack(q)[k])

            for ck, d in zip(np.unique(self.colors_12_from_0()[1]), (0, ySP)):
                cp0, c = self.colors_12_from_0()

                xp0, x = self.state_vector_x()
                x = self.reshape(self.unpack_color(x, c, ck), p0=None if cp0!=ck else xp0)

                yp0, y = self.state_vector_y()
                y = self.reshape(self.unpack_color(y, c, ck), p0=None if cp0!=ck else yp0)

                q_ = self.reshape(self.unpack_color(q, c, ck), p0=None if cp0!=ck else 0)

                if name=='v' and ck==1:
                    q_[k] += vSP

                if stitch:
                    if ck == 0:
                        x_last_slice = x[k][:, -1]
                        q_last_slice = q_[k][:, -1]
                    else:
                        # If the grid spacing is constant!!!
                        S = interpolation_matrix(self.fluid.u.y + d, self.fluid.u.y)

                        x[k] = np.hstack([x_last_slice.reshape((-1, 1)), x[k]])
                        y[k] = np.hstack([y[k][:,:1], y[k]])
                        q_[k] = np.hstack([(S@q_last_slice).reshape((-1, 1)), q_[k]])

                plt.pcolormesh(x[k], y[k] + d, q_[k], vmin=vmin, vmax=vmax, rasterized=True)

                if repeat:
                    x[k] = np.vstack([x[k][-1, :], x[k], x[k][0, :]])
                    y[k] = np.vstack([y[k][-1, :] - Ly, y[k], y[k][0, :] + Ly])
                    q_[k] = np.vstack([q_[k][-1, :], q_[k], q_[k][0, :]])

                    plt.pcolormesh(x[k], y[k] + d - Ly, q_[k], vmin=vmin, vmax=vmax, rasterized=True)
                    plt.pcolormesh(x[k], y[k] + d + Ly, q_[k], vmin=vmin, vmax=vmax, rasterized=True)

                for sxk, syk in zip(x[3:], y[3:]):
                    plt.plot(sxk[0], syk[0] + d, zorder=5)
                    if repeat:
                        plt.plot(sxk[0], syk[0] + d - Ly, zorder=5)
                        plt.plot(sxk[0], syk[0] + d + Ly, zorder=5)

                if borders:
                    plt.hlines((self.fluid.y[0] + d, self.fluid.y[-1] + d), 
                               xmin=np.min(x[0]), xmax=np.max(x[0]),
                               colors='k', lw=2, zorder=10)

            if equal:
                plt.axis('equal')
            if colorbar:
                plt.colorbar()

            plt.xlabel('x')
            plt.ylabel('y')

            if xlim:
                plt.xlim(*xlim)
            if ylim:
                plt.ylim(*ylim)

            if borders:
                plt.vlines(self.xSP,
                           ymin=np.min(self.fluid.y), ymax=np.max(self.fluid.y),
                           colors='k', lw=2, linestyles='dashdot', zorder=10, label='S.P.')

        plt.tight_layout()


    def unpack_color(self, q, colors, color):
        """ Return flow variables associated with a given color """

        qu, qv, qp, *qs = self.unpack(q)
        cu, cv, cp, *cs = self.unpack(colors)

        u, v, p = qu[cu == color], qv[cv == color], qp[cp == color]

        s = []

        for sk, ck in zip(qs, cs):
            assert (ck[0][0] == ck[0]).all() and (ck[0][0] == ck[1]).all(), \
                "Each immersed boundary must have exactly one color."

            if ck[0][0] == color:
                s.append(sk)
        
        return (u, v, p, *s)


    def reshape(self, fields, p0=None, shapes=()):
        """Return reshaped unpacked state vector.

        Parameters
        ----------
        field : list
            Unpacked state vector.
        p0 : field[:].dtype
            Value of the pressure field at the pinned point
        shapes : tuple, optional
            List of shapes (default self.shapes()).

        Returns
        -------
        list of np.ndarray
            (u, v, p, ((f1, g1), (f2, g2), ...)) fields (reshaped)
        """

        if not shapes:
            shapes = [(self.fluid.u.shape[0], -1), 
                      (self.fluid.v.shape[0], -1), 
                      (self.fluid.p.shape[0], -1)]

        return super().reshape(fields, p0, shapes)



    def state_vector_name(self):
        u = np.array(('u',)*self.fluid.u.size).reshape(self.fluid.u.shape)
        v = np.array(('v',)*self.fluid.v.size).reshape(self.fluid.v.shape)
        p = np.array(('p',)*self.fluid.p.size).reshape(self.fluid.p.shape)
        p0 = 'p'

        s = [ (np.array((sk.name+'_fx',)*sk.l), 
               np.array((sk.name+'_fy',)*sk.l)) for sk in self.solids]

        return p0, super().pack(u, v, p, *s)


    def state_vector_x(self):
        u = np.meshgrid(self.fluid.u.x, self.fluid.u.y)[0]
        v = np.meshgrid(self.fluid.v.x, self.fluid.v.y)[0]
        p = np.meshgrid(self.fluid.p.x, self.fluid.p.y)[0]
        p0 = p.ravel()[self.pZero]
        
        s = [(sk.ξ,)*2 for sk in self.solids]
        return p0, super().pack(u, v, p, *s)


    def state_vector_y(self):
        u = np.meshgrid(self.fluid.u.x, self.fluid.u.y)[1]
        v = np.meshgrid(self.fluid.v.x, self.fluid.v.y)[1]
        p = np.meshgrid(self.fluid.p.x, self.fluid.p.y)[1]
        p0 = p.ravel()[self.pZero]
 
        s = [(sk.η,)*2 for sk in self.solids]
        return p0, super().pack(u, v, p, *s)


    def state_vector_j(self):
        u = np.meshgrid(np.arange(self.fluid.u.shape[1]), np.arange(self.fluid.u.shape[0]))[0]
        v = np.meshgrid(np.arange(self.fluid.v.shape[1]), np.arange(self.fluid.v.shape[0]))[0]
        p = np.meshgrid(np.arange(self.fluid.p.shape[1]), np.arange(self.fluid.p.shape[0]))[0]
        p0 = p.ravel()[self.pZero]
        
        s = [(np.arange(sk.l),)*2 for sk in self.solids]
        return p0, super().pack(u, v, p, *s)


    def state_vector_i(self):
        u = np.meshgrid(np.arange(self.fluid.u.shape[1]), np.arange(self.fluid.u.shape[0]))[1]
        v = np.meshgrid(np.arange(self.fluid.v.shape[1]), np.arange(self.fluid.v.shape[0]))[1]
        p = np.meshgrid(np.arange(self.fluid.p.shape[1]), np.arange(self.fluid.p.shape[0]))[1]
        p0 = p.ravel()[self.pZero]
        
        s = [(np.arange(sk.l),)*2 for sk in self.solids]
        return p0, super().pack(u, v, p, *s)


    def advection_(self, u, v, uBC, vBC, ySP, vSP):
        Su, Sv, SuBC, SvBC = self.sliding_plane_interpolation(u, v, uBC=uBC, vBC=vBC, ySP=ySP, vAdd=vSP)
        NSu, NSv = self.reshape(self.fluid.advection(Su, Sv, SuBC, SvBC))
        Nu, Nv = self.sliding_plane_interpolation(NSu, NSv, ySP=-ySP)
        return Nu, Nv


    def sliding_plane_interpolation(self, u, v, p=None, uBC=None, vBC=None, ySP=0, vAdd=0):
        # first u and uBC
        S = interpolation_matrix(self.fluid.u.y, self.fluid.u.y + ySP)

        Su = u.copy()
        for slidek in range(self.iuSP, self.fluid.u.shape[1]):
            Su[:, slidek] = S@u[:, slidek]
        if uBC is not None:
            SuBC = uBC.copy()
            SuBC[1] = S@uBC[1]
            
        # then v and vBC
        Sv = v.copy()
        for slidek in range(self.ivSP, self.fluid.v.shape[1]):
            Sv[:, slidek] = S@v[:, slidek] + vAdd

        if vBC is not None:
            SvBC = vBC.copy()
            SvBC[1] = S@vBC[1] + vAdd

        # and finally p
        if p is not None:
            Sp = p.copy()
            for slidek in range(self.ipSP, self.fluid.p.shape[1]):
                Sp[:, slidek] = S@p[:, slidek]
            
            if uBC is None:
                return Su, Sv, Sp
            else:
                return Su, Sv, Sp, SuBC, SvBC
        else:
            if uBC is None:
                return Su, Sv
            else:
                return Su, Sv, SuBC, SvBC


    def interpolation_matrices_update(self, ySP, *M):
        S = interpolation_matrix(self.fluid.u.y + ySP, self.fluid.u.y)

        for Mk in M:
            data = []
            for n, j in zip(Mk.S1name, Mk.S1j):
                data.append(S[1:, 1:].ravel() if j==self.pZero and n=='p' else S.ravel())

            if len(data)!=0:
                Mk.S1.data = np.concatenate(data)

        S = interpolation_matrix(self.fluid.u.y, self.fluid.u.y + ySP)

        for Mk in M:
            data = []
            for n, j in zip(Mk.S2name, Mk.S2j):
                data.append(S[1:, 1:].ravel() if j==self.pZero and n=='p' else S.ravel())

            if len(data)!=0:
                Mk.S2.data = np.concatenate(data)
