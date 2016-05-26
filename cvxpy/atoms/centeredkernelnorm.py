"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from cvxpy.atoms.atom import Atom
#from cvxpy.atoms.axis_atom import AxisAtom
import cvxpy.lin_ops.lin_utils as lu
import numpy as np
import scipy.sparse as sp
from cvxpy.utilities.power_tools import pow_high, pow_mid, pow_neg, gm_constrs
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.soc_axis import SOC_Axis
from fractions import Fraction

class centeredkernelnorm(Atom):
    r"""The centered kernel 2-norm.

    If given a kernel matrix, index i, and weight  variable, ``centeredkernelnorm`` will treat it as the i-th vector in a Hilbert space, and compute the 2-norm of the i-the vector.

    The centered kernel 2-norm is given by

    .. math::

        \|\phi(x_i) - \mu\|_2 = K_{ii} - 2 K_{:,i}^T w + w^T K w,

    where :math:`\mu = Xw = \sum_i w_i x_i`,

    with domain :math:`w \in \mathbf{R}^n, K \in \mathbf{R}^{n \times n}, i \in [n], \phi(x_i) \in \mathbf{H}`, and :math:`\mathbf{H}` is the Hilbert space induced by the Gram matrix :math:`K`.


    Parameters
    ----------
		K : numpy matrix
				The kernel Gram matrix

    i : int
				The index of the vector for which we are computing the norm, ie :math:`K_{ii} = \phi(x_i)^T\phi(x_i)`.

    w : cvxpy.Variable
        The weight vector determining :math:`\mu \in \mathbf{H}`

    Returns
    -------
    Expression
        An Expression representing the norm.
    """
    def __init__(self, K, i, w):
				super(centeredkernelnorm).__init__(K,i,w)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the centered-kernel-norm of x_i.
        """
				K = self.args[0]
				i = self.args[1]
				w = self.args[2]
				return np.sqrt(K[i,i] - 2*np.dot(K[i,:],w.value) + np.dot(w.value.T, np.dot(K, w.value))

    def validate_arguments(self):
        super(pnorm, self).validate_arguments()
				K = self.args[0]
				i = self.args[1]
				w = self.args[2]
        if K.shape[1] != w.shape[0]
            raise ValueError(
                "K and w must have the same inside shape for matrix-vector multiplication.")
        if i < 0 or K.shape[0] <= i or K.shape[1] <= i
            raise ValueError(
                "i must be in the range of the size of K.")

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
        return (True, False)

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return True 

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return False

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return True # self.args[0].is_positive()

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return False #self.p >= 1 and self.args[0].is_negative()

    def get_data(self):
        return [self.K, self.i, self.w]

    def name(self):
        return "%s(K,i,%s)" % (self.__class__.__name__,
                               w.name())
    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        return []

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        return None #self._axis_grad(values)

    def _column_grad(self, value):
        """Gives the (sub/super)gradient of the atom w.r.t. a column argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            value: A numeric value for a column.

        Returns:
            A NumPy ndarray matrix or None.
        """
				return None
        #rows = self.args[0].size[0]*self.args[0].size[1]
        #value = np.matrix(value)
        ## Outside domain.
        #if self.p < 1 and np.any(value <= 0):
        #    return None
        #D_null = sp.csc_matrix((rows, 1), dtype='float64')
        #if self.p == 1:
        #    D_null += (value > 0)
        #    D_null -= (value < 0)
        #    return sp.csc_matrix(D_null.A.ravel(order='F')).T
        #denominator = np.linalg.norm(value, float(self.p))
        #denominator = np.power(denominator, self.p - 1)
        ## Subgrad is 0 when denom is 0 (or undefined).
        #if denominator == 0:
        #    if self.p >= 1:
        #        return D_null
        #    else:
        #        return None
        #else:
        #    nominator = np.power(value, self.p - 1)
        #    frac = np.divide(nominator, denominator)
        #    return np.reshape(frac.A, (frac.size, 1))

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        r"""Reduces the atom to an affine expression and list of constraints.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        size : tuple
            The size of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)

        Notes
        -----

        Implementation notes.

        - For general :math:`p \geq 1`, the inequality :math:`\|x\|_p \leq t`
          is equivalent to the following convex inequalities:

          .. math::

              |x_i| &\leq r_i^{1/p} t^{1 - 1/p}\\
              \sum_i r_i &= t.

          These inequalities happen to also be correct for :math:`p = +\infty`,
          if we interpret :math:`1/\infty` as :math:`0`.

        - For general :math:`0 < p < 1`, the inequality :math:`\|x\|_p \geq t`
          is equivalent to the following convex inequalities:

          .. math::

              r_i &\leq x_i^{p} t^{1 - p}\\
              \sum_i r_i &= t.

        - For general :math:`p < 0`, the inequality :math:`\|x\|_p \geq t`
          is equivalent to the following convex inequalities:

          .. math::

              t &\leq x_i^{-p/(1-p)} r_i^{1/(1 - p)}\\
              \sum_i r_i &= t.




        Although the inequalities above are correct, for a few special cases, we can represent the p-norm
        more efficiently and with fewer variables and inequalities.

        - For :math:`p = 1`, we use the representation

            .. math::

                x_i &\leq r_i\\
                -x_i &\leq r_i\\
                \sum_i r_i &= t

        - For :math:`p = \infty`, we use the representation

            .. math::

                x_i &\leq t\\
                -x_i &\leq t

          Note that we don't need the :math:`r` variable or the sum inequality.

        - For :math:`p = 2`, we use the natural second-order cone representation

            .. math::

                \|x\|_2 \leq t

          Note that we could have used the set of inequalities given above if we wanted an alternate decomposition
          of a large second-order cone into into several smaller inequalities.

        """
        p = data[0]
        axis = data[1]
        x = arg_objs[0]
        t = lu.create_var((1, 1))
        constraints = []

        # first, take care of the special cases of p = 2, inf, and 1
        if p == 2:
            if axis is None:
                return t, [SOC(t, [x])]

            else:
                t = lu.create_var(size)
                return t, [SOC_Axis(lu.reshape(t, (t.size[0]*t.size[1], 1)),
                                    x, axis)]

        if p == np.inf:
            t_ = lu.promote(t, x.size)
            return t, [lu.create_leq(x, t_), lu.create_geq(lu.sum_expr([x, t_]))]

        # we need an absolute value constraint for the symmetric convex branches (p >= 1)
        # we alias |x| as x from this point forward to make the code pretty :)
        if p >= 1:
            absx = lu.create_var(x.size)
            constraints += [lu.create_leq(x, absx), lu.create_geq(lu.sum_expr([x, absx]))]
            x = absx

        if p == 1:
            return lu.sum_entries(x), constraints

        # now, we take care of the remaining convex and concave branches
        # to create the rational powers, we need a new variable, r, and
        # the constraint sum(r) == t
        r = lu.create_var(x.size)
        t_ = lu.promote(t, x.size)
        constraints += [lu.create_eq(lu.sum_entries(r), t)]

        # make p a fraction so that the input weight to gm_constrs
        # is a nice tuple of fractions.
        p = Fraction(p)
        if p < 0:
            constraints += gm_constrs(t_, [x, r],  (-p/(1-p), 1/(1-p)))
        if 0 < p < 1:
            constraints += gm_constrs(r,  [x, t_], (p, 1-p))
        if p > 1:
            constraints += gm_constrs(x,  [r, t_], (1/p, 1-1/p))

        return t, constraints

        # todo: no need to run gm_constr to form the tree each time. we only need to form the tree once
