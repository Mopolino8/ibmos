import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from .tools import solver_default, perm_from_colors, submats_from_colors, mat_from_perm, iperm, subvec_from_colors

class Matrix12:
    """ Helper class for linear operators in sliding domains. """

    def __init__(self, M, colorsi_12, colorsj_12, j=None, name=None):
        """Initialize operator.

        Parameters
        ----------
        M : (N, N) array_like
            Matrix.
        colorsi_12 : np.array
            Coloring of the rows.
        colorsj_12 : np.array
            Coloring of the columns.
        j : np.array, optional
            Index of the x coordinate of the column. Needed for matrix-vector products.
        name : np.array, optional
            Variable name of each column. Needed for matrix vector-products. 
            
        Note
        ----
        By default, the two domains are decoupled (S1 and S2 set to zero).
        These interpolation matrices must be filled by the user. """

        # Save colors_12 and permutation
        self.colorsi_12 = colorsi_12
        self.colorsj_12 = colorsj_12

        self.permi_12 = perm_from_colors(self.colorsi_12)
        self.permj_12 = perm_from_colors(self.colorsj_12)

        # Find colors_12s and save permutation
        self.M = submats_from_colors(M.tocsr(), colorsi_12, colorsj_12)

        # Save interpolation matrices structure
        if j is not None:
            self.S1, self.S1j, self.S1name = _interp_ij(self.M[1][0].tocoo(), 
                                                    subvec_from_colors(j, 0, self.colorsj_12),
                                                    subvec_from_colors(name, 0, self.colorsj_12))
            self.S2, self.S2j, self.S2name = _interp_ij(self.M[0][1].tocoo(), 
                                                    subvec_from_colors(j, 1, self.colorsj_12), 
                                                    subvec_from_colors(name, 1, self.colorsj_12))

    def dot(self, q_, S1=None, S2=None):
        """ Matrix-vector product.

        Parameters
        ----------
        q_ : 1d array_like
            Vector.
        S1 : 2d array_like, optional
            Interpolation of (x1, q1) at x2. Default self.S1.
        S2 : 2d array_like, optional
            Interpolation of (x2, q2) at x1. Default self.S2.
        
        Return
        ------
        1d array_like
            Matrix vector product. """
        
        S1 = self.S1 if S1 is None else S1
        S2 = self.S2 if S2 is None else S2

        q = self.unpack(q_)
        r1 = self.M[0][0]@q[0]    + self.M[0][1]@S2@q[1]
        r2 = self.M[1][0]@S1@q[0] + self.M[1][1]@q[1]
        return self.pack(r1, r2)

    def __matmul__(self, q_):
        """ Same as self.dot with the default parameters. """

        return self.dot(q_)

    def dotT(self, r_, S1=None, S2=None):
        """ Matrix.T-vector product.

        Parameters
        ----------
        r_ : 1d array_like
            Vector.
        S1 : 2d array_like, optional
            Interpolation of (x1, q1) at x2. Default self.S1.
        S2 : 2d array_like, optional
            Interpolation of (x2, q2) at x1. Default self.S2.
        
        Return
        ------
        1d array_like
            Matrix vector product. """

        S1 = self.S1 if S1 is None else S1
        S2 = self.S2 if S2 is None else S2

        r = np.split(r_[self.permi_12], (self.M[0][0].shape[0],))

        q1 =      self.M[0][0].T@r[0] + S1.T@self.M[1][0].T@r[1]
        q2 = S2.T@self.M[0][1].T@r[0] +      self.M[1][1].T@r[1]
        return np.r_[q1, q2][iperm(self.permj_12)]

    def unpack(self, q):
        """ Unpack q into [q1, q2] """

        return np.split(q[self.permj_12], (self.M[0][0].shape[1],)) 

    def pack(self, q1, q2):
        """ Pack q1, q2 into q. """
        return np.r_[q1, q2][iperm(self.permi_12)]

class PoincareSteklov:
    """ Poincare--Steklov domain decomposition technique."""

    def __init__(self, M, colors_12, j=None, name=None):
        """Initialize operator.

        Parameters
        ----------
        M : (N, N) array_like
            Matrix.
        colors_12 : np.array
            Coloring of columns and rows.
        j : np.array, optional
            Index of the x coordinate of the column. Needed for matrix-vector products.
        name : np.array, optional
            Variable name of each column. Needed for matrix vector-products. 
            
        Note
        ----
        By default, the two domains are decoupled (S1 and S2 set to zero).
        These interpolation matrices must be filled by the user. """

        # Save colors_12 and permutation
        self.colors_12 = colors_12
        self.perm_12 = perm_from_colors(self.colors_12)

        # Find colors_12s and save permutation
        [[M11, M12], [M21, M22]] = submats_from_colors(M.tocsr(), colors_12)

        p12, p21 = np.zeros(M12.shape[0], dtype=bool), np.zeros(M21.shape[0], dtype=bool) 
        p12[np.unique(M12.tocoo().row)], p21[np.unique(M21.tocoo().row)] = True, True

        self.colors_12s = 1*np.r_[np.zeros_like(p12), ~p21] + 2*np.r_[p12, p21]
        self.perm_12s = perm_from_colors(self.colors_12s)

        # Find colors_12s_0 and save permutation
        self.colors_12s_0 = self.colors_12s[iperm(self.perm_12)]
        self.perm_12s_0 = self.perm_12[self.perm_12s]

        # Save interpolation matrices structure
        if j is not None:
            self.S1, self.S1j, self.S1name = _interp_ij(M21.tocoo(), 
                                                    subvec_from_colors(j, 0, self.colors_12), 
                                                    subvec_from_colors(name, 0, self.colors_12))
            self.S2, self.S2j, self.S2name = _interp_ij(M12.tocoo(), 
                                                subvec_from_colors(j, 1, self.colors_12), 
                                                subvec_from_colors(name, 1, self.colors_12))

        # Save invariant part M, Mss, Mss_12, Mss_21, Ps2, Ps1

        # Permutation 12 -> 12s
        [[P11, P12], [P21, P22], [Ps1, Ps2]] = submats_from_colors(mat_from_perm(self.perm_12s), 
                                          self.colors_12s[self.perm_12s], self.colors_12[self.perm_12])

        assert (P12.nnz == 0) and (P21.nnz == 0), "Permutation looks wrong!"
        
        self.M = [[P11@M11@P11.T,          None, P11@M11@Ps1.T], 
                  [         None, P22@M22@P22.T, P22@M22@Ps2.T],
                  [Ps1@M11@P11.T, Ps2@M22@P22.T,          None]]

        self.Mss = Ps1@M11@Ps1.T + Ps2@M22@Ps2.T

        self.Mss_12 = Ps1@M12
        self.Mss_21 = Ps2@M21

        self.Ps2 = Ps2
        self.Ps1 = Ps1

        self._factorized = False

    def factorize(self, exact = False, solver=solver_default()):
        """Precompute factorization.

        Parameters
        ----------
        exact: bool, optional
            Compute factorization without Poincare--Steklov. (default False)
        solver: callable, optional
            Solver.
            
        Note
        ----
        If exact is False (the default) factorization is computed as follows:

        M11 q1          + M1s qs = r1 -> q1 = iM11(r1 - M1s qs) 
                 M22 q2 + M2s qs = r2 -> q2 = iM22(r2 - M2s qs) 
        Ms1 q1 + Ms2 q2 + Mss qs = rs -> 
                (Mss - Ms1 iM11(M1s) - Ms2 iM22(M2s)) qs = 
                            rs - Ms1 iM11(r1) - Ms2 iM22(r2) 

        where M11, M12, M22, Ms1, Ms2, M1s, M2s and Mss are computed
        using the 12s<->12 permutation. S1 and S2 come in the calculation
        of Mss, whose factorization is postponed to solve() calls.

        The exact option is provided only for debugging purposes.
        
        """

        self._factorized = False
        self._exact = exact

        if self._exact:
            self.iM = solver(sp.bmat(self.matrix()).tocsc())[0]
        else:
            M11, M1s = self.M[0][0], self.M[0][2]
            M22, M2s = self.M[1][1], self.M[1][2]
            Ms1, Ms2 = self.M[2][0], self.M[2][1]

            self.iM11, self.iM22 = solver(M11.tocsc())[0], solver(M22.tocsc())[0]

            # Ms1_iM11_M1s
            tmp1 = sp.lil_matrix((M11.shape[0], M1s.shape[1]))
            for col in np.unique(M1s.tocoo().col):
                tmp1[:, col] = self.iM11(M1s[:, col].todense())
                err = la.norm(M11@tmp1[:,col].todense() - M1s[:,col].todense())/la.norm(M1s[:,col].todense())
                if err >= 1e-12:
                    print (f"Schur complement Ms1_iM11_M1s might not be accurate enough {err}")

            tmp1 = Ms1@tmp1.tocsr()

            # Ms2_iM22_M2s
            tmp2 = sp.lil_matrix((M22.shape[0], M2s.shape[1]))
            for col in np.unique(M2s.tocoo().col): 
                tmp2[:, col] = self.iM22(M2s[:, col].todense())
                err = la.norm(M22@tmp2[:,col].todense() - M2s[:,col].todense())/la.norm(M2s[:,col].todense())
                if err >= 1e-12:
                    print (f"Schur complement Ms2_iM22_M2s might not be accurate enough {err}")

            tmp2 = Ms2@tmp2.tocsr()

            self.iMss = self.Mss - tmp1 - tmp2

        self._factorized = True

    def matrix(self, S1=None, S2=None):
        """ Retrieve matrix with the interpolation.

        Parameters
        ----------
        S1 : 2d array_like, optional
            Interpolation of (x1, q1) at x2. Default self.S1.
        S2 : 2d array_like, optional
            Interpolation of (x2, q2) at x1. Default self.S2.
    
        Return
        ------
        array_like
            Matrix. """


        S1 = self.S1 if S1 is None else S1
        S2 = self.S2 if S2 is None else S2

        M = self.M

        M[2][2] = self.Mss + self.Mss_21@S1@self.Ps1.T + self.Mss_12@S2@self.Ps2.T

        return M

    def dot(self, q_, S1=None, S2=None):
        """ Matrix-vector product.

        Parameters
        ----------
        q_ : 1d array_like
            Vector.
        S1 : 2d array_like, optional
            Interpolation of (x1, q1) at x2. Default self.S1.
        S2 : 2d array_like, optional
            Interpolation of (x2, q2) at x1. Default self.S2.
        
        Return
        ------
        1d array_like
            Matrix vector product. """

        q = self.unpack(q_)

        M = self.matrix(S1, S2)

        r1 = M[0][0]@q[0] + M[0][2]@q[2]
        r2 =                M[1][1]@q[1] + M[1][2]@q[2]
        rs = M[2][0]@q[0] + M[2][1]@q[1] + M[2][2]@q[2]

        return self.pack(r1, r2, rs)

    def __matmul__(self, q_):
        """ Same as self.dot with the default parameters. """

        return self.dot(q_)
    
    def solve(self, r_, S1=None, S2=None, q0_ = None, solver=solver_default()):
        """ Solve M q = r 

        Parameters
        ----------
        q_ : 1d array_like
            Vector.
        S1 : 2d array_like, optional
            Interpolation of (x1, q1) at x2. Default self.S1.
        S2 : 2d array_like, optional
            Interpolation of (x2, q2) at x1. Default self.S2.
        q0_ : 1d array_like
            Initial guess
        solver: callable, optional
            Solver.
        
        Return
        ------
        1d array_like
            Solution q. """

        r = self.unpack(r_)

        S1 = self.S1 if S1 is None else S1
        S2 = self.S2 if S2 is None else S2

        q0 = None if q0_ is None else self.unpack(q0_)

        # M11 q1          + M1s qs = r1 -> q1 = iM11(r1 - M1s qs) 
        #          M22 q2 + M2s qs = r2 -> q2 = iM22(r2 - M2s qs) 
        # Ms1 q1 + Ms2 q2 + Mss qs = rs -> (Mss - Ms1 iM11(M1s) - Ms2 iM22(M2s)) qs = rs - Ms1 iM11(r1) - Ms2 iM22(r2) 

        if not self._factorized:
            self.factorize(solver)

        if self._exact:
            q = self.iM(np.concatenate(r), x0 = None if q0 is None else np.concatenate(q0))
            [q1, q2, qs]  = np.split(q, (len(r[0]), len(r[0])+len(r[1])))
        else:
            r_hat = r[2] - self.M[2][0]@self.iM11(r[0]) - self.M[2][1]@self.iM22(r[1]) 

            iMss = self.iMss + self.Mss_21@S1@self.Ps1.T + self.Mss_12@S2@self.Ps2.T

            #It seems the dense solver is faster!
            #iMss = solver(self.iMss.tocsc())[0]
            #qs = iMss(r_hat, x0 = q0[2] if q0 else None)

            qs = la.solve(iMss.todense(), r_hat)

            q1 = self.iM11(r[0] - self.M[0][2]@qs, x0 = q0[0] if q0 else None)
            q2 = self.iM22(r[1] - self.M[1][2]@qs, x0 = q0[1] if q0 else None)

        return self.pack(q1, q2, qs)

    def pack(self, q1, q2, qs):
        """ Unpack q into [q1, q2, qs] """

        return np.r_[q1, q2, qs][iperm(self.perm_12s_0)]

    def unpack(self, q):
        """ Pack [q1, q2, qs] into q """

        i1 = self.M[0][0].shape[0]
        i2 = self.M[1][1].shape[0] + i1

        return np.split(q[self.perm_12s_0], [i1, i2]) 

def _interp_ij(Mij, jj, nj, verbose=False):
    """ Determine the structure of the interpolation operators.

        Parameters
        ----------
        Mij : array_like
            Matrix for the interaction between color i and color j
        jj : array_like
            Vector with the slice (x coord constant) associated with each column
        nj : array_like
            Vector with the variable name associated with each column.
        verbose : boo, optional
            Print extra information.
        
        Return
        ------
        S : sparse matrix
            Empty matrix with the structure of the interpolation ij operator.
        j : list
            List of slices.
        n : list
            List of names."""
    jnz, n = np.unique(Mij.col), Mij.shape[1]

    Si, Sj = [], []

    names, j = [], []

    if verbose:
        print('------')

    if len(jnz)!=0:
        for name, slice_str in np.unique(np.vstack([nj[jnz], jj[jnz]]), axis=1).T:
            if name not in ('u', 'v', 'p'):
                raise ValueError(f"Unsupported case. (got name={name})")
            
            if verbose:
                print(name, slice_str)

            slice_ = int(slice_str)

            rows = np.arange(n)[(jj==slice_)*(nj==name)]
            Si.append(np.repeat(rows, len(rows)))
            Sj.append(np.concatenate([rows,]*len(rows)))
            
            names.append(name)
            j.append(slice_)

        Si = np.concatenate(Si)
        Sj = np.concatenate(Sj)

    Sdata = np.zeros_like(Si)
    S = sp.coo_matrix((Sdata, (Si, Sj)), (n, n))

    return S, j, names