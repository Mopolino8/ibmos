from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from .flow import Field
from .tools import solver_default

class Solver:
    """Flow solver based on the Projection-based Immersed Boundary Method.

    Attributes
    ----------
    fluid : Field
        Flow field information.
    solids : list
        List of immersed boundaries.

    """

    field: Field
    solids: list = []
    solver = None

    def __init__(self, x, y, iRe=1.0, Co=0.5, fractionalStep=False, solver=solver_default(), *solids):
        """Initialize solver.

        Parameters
        ----------
        x : np.array
            Coordinates of the vertices (x direction).
        y : np.array
            Coordinates of the vertices (y direction).
        iRe : float, optional
            Inverse of the Reynolds number.
        Co : float, optional
            Courant number.
        fractionalStep : bool, optional
            Fractional Step Method flag.
        solver : callable, optional
            `solver(A)` that returns linear solver.
        solids : list, optional
            List of solids. """

        # Create field and determine smallest cell size.
        self.fluid = Field(x, y)
        self.dxmin = np.min(np.r_[self.fluid.p.dx, self.fluid.p.dy])

        # Laplacian and divergence operators.
        self.laplacian = self.fluid.laplacian()
        self.divergence = self.fluid.divergence()

        # The state vector is formed by stacking u, v, p and if we have
        # solids, also by (fx, fy) as many times as the number of solids.
        # pStart points to the first pressure value in this state vector
        # pEnd points to the first value after the last pressure value.
        self.pStart = self.fluid.u.size + self.fluid.v.size
        self.pEnd = self.pStart + self.fluid.p.size - 1  # First value not included
        
        # Set parameters
        self.set_iRe(iRe)
        self.set_Co(Co)
        self.set_fractional_step(fractionalStep)
        self.set_solver(solver)
        
        self.set_solids(*solids)
        
    def cleanup(self):
        """Clean-up structures that must be recomputed after calls to set_*."""
        
        self.A, self.B, self.iA = None, None, None
        self.stepsInitialized = False
        
    def set_iRe(self, iRe):
        """Set inverse of the Reynolds number.
        
        Parameters
        ----------
        iRe : float
            Inverse of the Reynolds number.
            
        """
        self.iRe = iRe
        self.cleanup()
 

    def set_Co(self, Co):
        """Set courant number.
        
        Parameters
        ----------
        Co : float
            Courant number.
        """
        
        self.dt = Co * min(self.dxmin ** 2 / self.iRe, self.dxmin)
        self.cleanup()


    def set_fractional_step(self, fractionalStep):
        """Set fractional step method flag.

        Parameters
        ----------
        fractionalStep : bool
            Fractional step method flag.
            
        """
        
        self.fractionalStep = fractionalStep
        self.cleanup()
        
    def set_solver(self, solver):
        """Set linear solver.

        Parameters
        ----------
        solver : callable.
            Linear solver.
            
        """
        
        self.solver = solver
        self.cleanup()


    def set_solids(self, *solids):
        """Set immersed boundaries (solids).

        Parameters
        ----------
        solids : list
            List of Solid objects.

        Note
        ----
        The interest in specifying separate solids instead of one
        is that they get forces computed separately.
        """

        self.solids = solids
        self.E = []

        for solid in self.solids:
            Eu = solid.interpolation(self.fluid.u)
            Ev = solid.interpolation(self.fluid.v)
            self.E.append((Eu, Ev))
            
        self.cleanup()


    def jacobian(self, uBC, vBC, u0=None, v0=None):
        """Return Jacobian.

        Return Jacobian at `u0` and `v0`. If they are not provided, advection terms are
        not included.

        Parameters
        ----------
        u0 : np.ndarray, optional
            Horizontal velocity component at the linearization point.
        v0 : np.ndarray, optional
            Vertical velocity component at the linearization point.
        uBC : list
            Boundary conditions on the horizontal velocity component (uW, uE, uS, uN).
        vBC : list
            Boundary conditions on the vertical velocity component (vW, vE, vS, vN).

        Returns
        -------
            Jacobian in sparse-matrix form.
        """

        # Laplacian for u and v.
        L = sp.block_diag((self.laplacian[0][0], self.laplacian[1][0]))

        # Divergence-free constraint plus velocity boundary condition on
        # immersed boundaries (if needed).
        Q = [-sp.hstack((self.divergence[0][0], self.divergence[1][0]), format='csr') ]

        # Suppress first row -> set value of the pressure to zero at the first node.
        Q[0] = Q[0][1:, :]

        if self.solids:
            for Eu, Ev in self.E:
                Q.append(sp.block_diag((Eu, Ev), format='csr'))
        Q = sp.vstack(Q)

        Z = sp.coo_matrix((Q.shape[0],) * 2)

        # If u0 and v0 are provided, the Jacobian includes advection terms.
        if u0 is not None and v0 is not None:
            N = self.fluid.linearized_advection(u0, v0, uBC, vBC)
            J = sp.bmat([[-self.iRe * L + N, Q.T], [Q, Z]]).tocsc()
        else:
            J = sp.bmat([[-self.iRe * L, Q.T], [Q, Z]]).tocsc()

        return J

    def propagator(self, fractionalStep):
        """Return propagator.

        fractionalStep: bool, optional
            Propagators for fractional step method.

        Returns
        -------
        tuple:
            Propagator matrices.

        """
        Mu, Mv = self.fluid.u.weight_width(), self.fluid.v.weight_height()
        Ru, Rv = self.fluid.u.weight_height(), self.fluid.v.weight_width()

        # Mass matrix for u and v.
        M = sp.block_diag((Mu @ Ru, Mv @ Rv))

        # Laplacian for u and v.
        L = sp.block_diag((self.laplacian[0][0], self.laplacian[1][0]))

        # Divergence-free constraint plus velocity boundary condition on
        # immersed boundaries (if needed).
        Q = [-sp.hstack((self.divergence[0][0], self.divergence[1][0]), format='csr')]

        # Suppress first row -> set value of the pressure to zero at the first node.
        Q[0] = Q[0][1:, :]

        if self.solids:
            for Eu, Ev in self.E:
                Q.append(sp.block_diag((Eu, Ev), format='csr'))
        Q = sp.vstack(Q)

        Z = sp.coo_matrix((Q.shape[0],) * 2)

        A = (M / self.dt - 0.5 * self.iRe * L).tocsc()
        B = (M / self.dt + 0.5 * self.iRe * L).tocsr()
        BB = sp.block_diag((B, Z)).tocsr()

        if fractionalStep:
            iM = M.copy()
            iM.data[:] = 1 / M.data[:]

            iML = iM @ L

            BN = self.dt*iM + (0.5*self.iRe)*self.dt**2*iML@iM + (0.5*self.iRe)**2*self.dt**3*iML@(iML@iM)

            QBNQT = (Q @ (BN @ Q.T)).tocsc()

            return (A, QBNQT), (B, BN, Q)
        else:
            AA = sp.bmat([[A, Q.T], [Q, Z]]).tocsc()

            return (AA,), (BB,)


    def boundary_condition_terms(self, uBC, vBC, *sBC):
        """Return contribution of the boundary terms to the right-hand-side.

        Parameters
        ----------
        uBC : list
            List of np.ndarray vectors with the West, East, South and North
            boundary conditions for the horizontal component of the velocity
            at the time level t+1.
        vBC : list
            List of np.ndarray vectors with the West, East, South and North
            boundary conditions for the vertical component of the velocity
            at the time level t+1.
        sBC : list, optional
            List of np.ndarray with the horizontal and vertical component of
            the velocity on the immersed boundaries at the time level t+1.

        Returns
        -------
        np.ndarray
            Contribution of the boundary terms to the right-hand-side of tue
            governing equations..

        """

        bu = np.sum([self.iRe * A @ x for A, x in zip(self.laplacian[0][1], uBC)], axis=0)
        bv = np.sum([self.iRe * A @ x for A, x in zip(self.laplacian[1][1], vBC)], axis=0)

        # RHS terms for the divergence eq. except for the first cell (pressure set to zero)
        bD = (np.sum([A @ x for A, x in zip(self.divergence[0][1], uBC[:2])], axis=0) +
              np.sum([A @ x for A, x in zip(self.divergence[1][1], vBC[2:])], axis=0))[1:]

        bc = [bu, bv, bD]
        for sBCk in sBC:
            bc.append(np.r_[sBCk])

        return np.concatenate(bc)


    def steady_state(self, x0, uBC, vBC, sBC=(), outflowEast=False, xtol=1e-8, ftol=1e-8,
                     maxit=15, verbose=True, checkJacobian=False):
        """Compute steady state solution using exact Newton-Raphson iterations.

        Parameters
        ----------
        x0 : np.ndarray
            Initial guess (packed state-vector).
        uBC : list
            List of np.ndarray vectors with the West, East, South and North
            boundary conditions for the horizontal component.
        vBC : list
            List of np.ndarray vectors with the West, East, South and North
            boundary conditions.
        sBC : list, optional
            List of np.ndarray with the horizontal and vertical component of
            the velocity on the immersed boundaries.
        outflowEast: bool, optional
            Specify if the East boundary has an outflow boundary condition.
        xtol : float, optional
            Tolerance on the solution |x^{k+1} - x^k|_2/|x^k|_2.
        ftol : float, optional
            Tolerance on the function |f^{k+1} - f^k|_2/|b|_2.
        maxit : int, optional
            Maximum number of iterations.
        verbose : bool, optional
            Enable verbose output. Output includes iteration count, residuals
            and forces on the immersed boundaries.
        checkJacobian : bool, optional
            Check Jacobian against numerical approximation.

        Returns
        -------
        x : np.ndarray
            Result of the last iteration.
        infodict : dict
            Information on the performed iterations.

        """

        # Contribution of the boundary conditions to the right-hand-side
        bc = self.boundary_condition_terms(uBC, vBC, *sBC)
        # Build Jacobian without advection terms.
        JnoAdv = self.jacobian(uBC, vBC)

        # Copy state vector.
        x = x0.copy()

        # Dictionary with output variables
        header = ['residual_x', 'residual_f']
        header.extend(chain(*[(f'{solid.name}_fx', f'{solid.name}_fy')
                              for solid in self.solids]))

        # Create dictionary
        infodict = dict(zip(header, ([] for _ in header)))

        # If verbose, print header.
        if verbose:
            print("   k", "".join((f'{elem:>12} ' for elem in header)))


        # Newton-Raphson iterations
        for k in range(maxit):
            # Build right-hand-side vector: bc + advection terms.
            b = bc.copy()

            u0, v0 = self.reshape(*self.unpack(x))[:2]
            b[:self.pStart] -= np.r_[self.fluid.advection(u0, v0, uBC, vBC)]

            # Compute residual vector
            residual = JnoAdv @ x - b

            # Compute next estimate of the solution and subtract the average pressure.
            # The linear system is solved using direct methods.
            J = self.jacobian(uBC, vBC, u0, v0)

            if checkJacobian:
                h = 1e-8
                xtmp = x + 1j * h * np.random.random(x.shape)

                btmp = np.asarray(bc, dtype=xtmp.dtype)
                u0tmp, v0tmp = self.reshape(*self.unpack(xtmp))[:2]

                btmp[:self.pStart] -= np.r_[self.fluid.advection(u0tmp, v0tmp, uBC, vBC)]

                residualtmp = JnoAdv @ xtmp - btmp

                e1 = residualtmp.imag / h
                e2 = J @ ((xtmp - x).imag / h)
                eerr = la.norm(e1 - e2) / la.norm(e1)

                if ftol <= eerr:
                    print("Warning: Jacobian might not be accurate enough (eerr=%12e)" % eerr)

            xp1 = x - self.solver(J)[0](residual, x0=None if k==0 else x-xp1)  # Time consuming.

            # How much has the solution changed? How close is f(x^{k+1}) to zero?
            xp1mx = xp1 - x
            
            infodict['residual_x'].append(la.norm(xp1mx) / la.norm(xp1))
            infodict['residual_f'].append(la.norm(residual) / la.norm(b))
            
            if self.solids:
                fp1 = self.unpack(xp1)[3:]
                for l, solid in enumerate(self.solids):
                    infodict[f'{solid.name}_fx'].append(2*np.sum(fp1[2*l]))
                    infodict[f'{solid.name}_fy'].append(2*np.sum(fp1[2*l+1]))
                    
            # Print (if verbose) the iteration count, residuals and forces
            # on the immersed boundaries.
            if verbose:
                print(f"{k+1:4}", "".join((f'{infodict[elem][k]: 12.5e} ' for elem in header)))

            x = xp1

            # Refresh East boundary condition using average upstream the boundary.
            # WARNING:
            #     This may lead to diverging iterations if the boundary is not
            #     sufficiently far downstream from the obstacle(s).
            if outflowEast:
                u, v = self.reshape(*self.unpack(x))[:2]
                uBC[1][:], vBC[1][:] = np.mean(u[:, -5:], axis=1), np.mean(v[:, -5:], axis=1)
                bc = self.boundary_condition_terms(uBC, vBC, *sBC)

            # If the tolerances are reached, stop iterating.
            if infodict['residual_x'][-1] < xtol and infodict['residual_f'][-1] < ftol:
                break
        else:
            if verbose:
                print("Warning: maximum number of iterations reached (maxit=%d)" % maxit)

        infodict.update((key, np.asarray(value)) for key, value in infodict.items())

        return x, infodict


    def steps(self, x, uBC, vBC, sBC=(), outflowEast=False, number=1, saveEvery=None, 
              verbose=1, checkSolvers=False, Nm1=None):
        """Time-step the governing equations.

        Parameters
        ----------
        x : np.ndarray
            Initial condition (packed state-vector).
        uBC : list
            List of np.ndarray vectors with the West, East, South and North
            boundary conditions for the horizontal component.
        vBC : list
            List of np.ndarray vectors with the West, East, South and North
            boundary conditions.
        sBC : list, optional
            List of np.ndarray with the horizontal and vertical component of
            the velocity on the immersed boundaries.
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
        Nm1 : np.ndarray, optional
            Advection terms at the previous time-step. If None, the first
            step is performed using explicit Euler method.

        Returns
        -------
        xres : (np.ndarray)
            Flow fields sampled every saveEvery steps.
        tres: list (np.ndarray)
            Time.
        infodict: dict
            Norm of the state vector, temporal derivative, and forces.

        """
        if not self.stepsInitialized:
            self.A, self.B = self.propagator(self.fractionalStep)
            self.iA = [self.solver(Ak)[0] for Ak in self.A]
            self.stepsInitialized = True

        if saveEvery is None:
            saveEvery = number

        xres, tres = [], []

        # Advection terms at the CURRENT time step.
        N = np.r_[self.fluid.advection(*self.reshape(*self.unpack(x))[:2], uBC, vBC)]

        # If we were not provided with the advection terms at the PREVIOUS
        # time step, we use the current ones.
        if not Nm1:
            Nm1 = N

        # Contribution of the boundary conditions to the right-hand-side.
        bc = self.boundary_condition_terms(uBC, vBC, *sBC)

        # Dictionary with output variables
        header = ['t', 'x_2', 'dxdt_2']
        header.extend(chain(*[(f'{solid.name}_fx', f'{solid.name}_fy')
                              for solid in self.solids]))
        if outflowEast:
            header.append('Uinf@outlet')
        if checkSolvers:
            if self.fractionalStep:
                header.extend(['rel.error(A)', 'rel.error(C)'])
            else:
                header.append('rel.error(A)')

        # Create dictionary
        infodict = dict(zip(header, (np.empty(number) for _ in header)))

        # If verbose, print header.
        if verbose:
            print("       k", "".join((f'{elem:>12} ' for elem in header)))


        # Main loop.
        for k in range(number):
            infodict['t'][k] = (k+1)*self.dt

            # Build right-hand-side.
            # terms at current time step plus boundary conditions plus advection.
            # And compute timestep

            # Compute next time step. Time consuming part
            if self.fractionalStep:
                b = self.B[0] @ x[:self.pStart] + bc[:self.pStart]
                b += -1.5 * N + 0.5 * Nm1

                qast = self.iA[0](b, x0=None if k==0 else qast)
                λ = self.iA[1](self.B[2]@qast - bc[self.pStart:], x0=None if k==0 else λ)

                xp1 = np.r_[qast - self.B[1]@(self.B[2].T@λ), λ]

                if checkSolvers:
                    infodict['rel.error(A)'][k] = la.norm(self.A[0]@qast - b)/la.norm(b)
                    infodict['rel.error(C)'][k] = la.norm(self.A[1]@λ - self.B[2]@qast + bc[self.pStart:])/la.norm(self.B[2]@qast - bc[self.pStart:])
            else:
                b = self.B[0] @ x + bc
                b[:self.pStart] += -1.5 * N + 0.5 * Nm1
                xp1 = self.iA[0](b, x0=None if k==0 else xp1)

                if checkSolvers:
                    infodict['rel.error(A)'][k] = (la.norm(self.A[0]@xp1 - b)/la.norm(b))

            infodict['x_2'][k] = la.norm(xp1)
            infodict['dxdt_2'][k] = la.norm(xp1-x)/self.dt

            if self.solids:
                fp1 = self.unpack(xp1)[3:]
                for l, solid in enumerate(self.solids):
                    infodict[f'{solid.name}_fx'][k] = 2*np.sum(fp1[2*l])
                    infodict[f'{solid.name}_fy'][k] = 2*np.sum(fp1[2*l+1])

            if outflowEast:
                u, v = self.reshape(*self.unpack(xp1))[:2]
                
                Uinf = np.sum(self.fluid.u.dy*u[:,-1])/(self.fluid.y[-1]-self.fluid.y[0])
                infodict['Uinf@outlet'][k] = Uinf
                dx = (self.fluid.x[-1]-self.fluid.x[-2])
                uBC[1][:] = uBC[1][:] - Uinf*self.dt/dx*(uBC[1][:] - u[:,-1])
                vBC[1][:] = vBC[1][:] - Uinf*self.dt/dx*(vBC[1][:] - v[:,-1])
                bc = self.boundary_condition_terms(uBC, vBC, *sBC)

            # If reportEvery is not None, print current step, time, residuals and,
            # if we have immersed boundaries, print also the forces.
            if verbose and ((k + 1) % verbose == 0 or (k + 1) == number):
                print(f"{k+1:8}", "".join((f'{infodict[elem][k]: 12.5e} ' for elem in header)))

            # Prepare for the next time step
            x = xp1
            Nm1 = N
            if k != number - 1:
                N = np.r_[self.fluid.advection(*self.reshape(*self.unpack(x))[:2], uBC, vBC)]

            # Append vector to xres?
            if (k + 1) % saveEvery == 0:
                xres.append(x)
                tres.append((k+1)*self.dt)

        # Return state vectors
        return np.squeeze(xres), np.squeeze(tres), infodict

    def shapes(self):
        """Return the shapes of the fields stacked in the state vector.

        Returns
        -------
        list
            (u.shape, v.shape, p.shape, [(l1, l1), (l2, l2), ...])

        """
        shapes = [self.fluid.u.shape, self.fluid.v.shape, self.fluid.p.shape]

        if self.solids:
            for solid in self.solids:
                shapes.extend((solid.l,) * 2)

        return shapes

    def sizes(self):
        """Return the sizes of the fields stacked in the state vector.

        Returns
        -------
        list
            (u.size, v.size, p.size, [(l1, l1), (l2, l2), ...])

        """
        sizes = [self.fluid.u.size, self.fluid.v.size, self.fluid.p.size - 1]

        if self.solids:
            for solid in self.solids:
                sizes.extend((solid.l,) * 2)

        return sizes

    def zero_boundary_conditions(self):
        """Return zero boundary conditions."""
        uW, uE = np.zeros(self.fluid.u.shape[0]), np.zeros(self.fluid.u.shape[0])
        uS, uN = np.zeros(self.fluid.u.shape[1]), np.zeros(self.fluid.u.shape[1])
        uBC = (uW, uE, uS, uN)

        vW, vE = np.zeros(self.fluid.v.shape[0]), np.zeros(self.fluid.v.shape[0])
        vS, vN = np.zeros(self.fluid.v.shape[1]), np.zeros(self.fluid.v.shape[1])
        vBC = (vW, vE, vS, vN)

        return uBC, vBC

    def zero(self):
        """Return zero state vector (packed).

        Returns
        -------
        np.ndarray
            Array with zeros.

        """
        return np.zeros(np.sum(self.sizes()))

    def pack(self, u, v, p, *s):
        """Pack flow field and forces into a single vector.

        Parameters
        ----------
        u : np.ndarray
            Horizontal velocity field.
        v : np.ndarray
            Vertical velocity field.
        p : np.ndarray
            Pressure field.
        s : list, optional
            Forces, i.e. [(f1, g1), (f2, g2), ...], on the solids.

        Returns
        -------
        np.ndarray
            Concatenated fields (u, v, p, [f1, g1, f2, g2, ...]).
        """

        fields = [u.ravel(), v.ravel(), p.ravel()[1:]-p[0, 0]]

        for (f, g) in s:
            fields.append(f)
            fields.append(g)

        return np.concatenate(fields)

    def unpack(self, x):
        """Return unpacked state vector.

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        list of np.ndarray
            (u, v, p, [f1, g1, f2, g2, ...]) fields (ravelled)
        """
        return np.split(x, np.cumsum(self.sizes()[:-1]))

    def reshape(self, *fields):
        """Return reshaped unpacked state vector.

        Parameters
        ----------
        field : list
            Unpacked state vector.

        Returns
        -------
        list of np.ndarray
            (u, v, p, [f1, g1, f2, g2, ...]) fields (reshaped)
        """

        reshaped = []

        for k, (field, shape) in enumerate(zip(fields, self.shapes())):
            if k!=2:
                reshaped.append(field.reshape(shape))
            else:
                reshaped.append(np.r_[0, field].reshape(shape))

        return reshaped

    def plot_domain(self, equal=True, figsize=(6, 6), xlim=(), ylim=()):
        """Plot domain and immersed boundaries.
        
        Parameters
        ----------
        equal: bool, optional
            Use equal axes.
        figsize: optional, tuple
            Size of the figure.
        xlim: optional, tuple
            x limits.
        ylim: optional, tuple
            y limits.
        """

        fig = plt.figure(figsize=figsize)
        plt.title('Fluid domain and immersed boundaries')

        X, Y = np.meshgrid(self.fluid.x, self.fluid.y)
        plt.plot(X, Y, 'b-', lw=0.5);
        plt.plot(X.T, Y.T, 'b-', lw=0.5);
        for solid in self.solids:
            plt.plot(solid.ξ, solid.η, 'o-', label=solid.name);
        if equal:
            plt.axis('equal')
        if self.solids:
            plt.legend()

        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        if xlim:
            plt.xlim(*xlim)

        if ylim:
            plt.ylim(*ylim)

    def plot_field(self, x, colorbar=False, equal=True, figsize=(8, 3), xlim=(), ylim=()):
        """Plot field.

        Parameters
        ----------
        x: np.ndarray
            State vector (packed).
        colorbar: bool, optional
            Display colorbar.
        equal: bool, optional
            Use equal axes.
        figsize: optional, tuple
            Size of the figure.
        xlim: optional, tuple
            x limits.
        ylim: optional, tuple
            y limits.

        """
        
        u, v, p = self.reshape(*self.unpack(x))[:3]

        fig = plt.figure(figsize=figsize)

        plots = ((self.fluid.u.x, self.fluid.u.y, u, 'u'),
                 (self.fluid.v.x, self.fluid.v.y, v, 'v'),
                 (self.fluid.p.x, self.fluid.p.y, p, 'p'))

        for k, (x, y, f, fname) in enumerate(plots):
            plt.subplot(1, len(plots), k + 1)
            plt.title(fname)
            plt.pcolormesh(x, y, f, rasterized=True)
            for solid in self.solids:
                plt.plot(solid.ξ, solid.η)
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

        plt.tight_layout()
