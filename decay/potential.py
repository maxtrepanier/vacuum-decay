#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r""" Implements a toolbox for defining and evaluating a potential function. The
module defines the generic class Potential, as well as implementations for
specific classes of potentials: polynomial, gaussian. """

# Dependencies
import numpy as np
from scipy import integrate as integrate
from scipy import optimize as optimize
import warnings

__author__ = "Maxime Trépanier"
__date__ = "01/30/2017"
# 29/3/2024: comments added, compatibility with SciPy 1.12


def critical_ODE(x, E, v, k, d=4):
    """ ODE of the critical solution. x is the domain, E the energy, v the
    potential function, k , d the dimensionality of spacetime. """
    if E + v(x) < 0:
        return np.nan
    return (d-1) * k * (2 * E * (E + v(x)))**0.5


def toOptimize(k, v, v0p=1e-8):
    """ Finding the zero of this function (for a value of k) gives the critical
    k_star. This is the value of k s.t. the integral of critical_ODE has E = 0
    at both xF and xT.

    If k is too small, the solution will become complex before reaching xT, in
    which case toOptimize returns False. If k is too large, E(xT) > 0, and
    toOptimize returns E(xT).
    """
    def event_zero_energy(x, E):
        return E[-1] + v.v(x) - 1e-9
    event_zero_energy.terminal = True

    x = np.linspace(v.xF, v.xT, 100)
    ode_soln = integrate.solve_ivp(lambda x, E: critical_ODE(x, E, v=v.v, k=k),
            (v.xF, v.xT),  # range of integration
            [v0p],  # initial value
            t_eval=x,  # values to save
            events=event_zero_energy  # stop at E = -v
            )
    if ode_soln.status == 1:
        return -1  # k too small
    else:
        return ode_soln.y[0,-1]  # k too large


# Generic potential
class Potential():
    """ This class encapsulates a potential function and defines the various
    functions that are needed to solve the system of equations. """

    def __init__(self):
        r""" Constructor. The function computes the main reference points on v,
        namely xF, xT, xmax and xminE, respectively the false vacuum position,
        the true vacuum position, the maximum of the potential and the position
        where v(xminE) = v(xF), which is the minimal energy to have a complete
        transition. """

        self.xF = 0  # position of false vacuum
        self.xT = 0  # position of true vacuum
        self.xmax = 0  # maximum of the potential
        self.xminE = 0  # minimal value of E required for the instanton
        self.kc = None  # Critical value of k
        # range of values to search for kc
        self.kcmin = 1e-2
        self.kcmax = 0.8
        self.name = ""  # name, for identification

    def v(self, x, xi=0):
        r""" Returns the value of v(x), with an offset xi s. t.
        :math:`v(x_F, \xi = 0) = 0` and :math:`v(x_T, \xi = 1) = 0`.

        :param x: Scalar field
        :type x: float
        :param xi: False vacuum energy density
        :type xi: float
        :rtype: float
        """
        pass

    def dv(self, x):
        r""" Returns the value of v'(x).
        :param x: Scalar field
        :type x: float
        :rtype: float
        """
        pass

    def ddv(self, x):
        r""" Returns the value of v''(x).
        :param x: Scalar field
        :type x: float
        :rtype: float
        """
        pass

    def kstar(self, xi0=0):
        """ Saves the value of \kappa_c. Returns the value of \kappa_c,
        estimated from the positive energy theorem. """
        # Solving for kc
        if self.kc is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.kc = optimize.brentq(
                    lambda k: toOptimize(k, self, 1e-8), self.kcmin, self.kcmax, xtol=1e-10)

        # Positive energy thm
        if xi0 == 0:
            return self.kc
        else:
            dv = self.v(self.xF) - self.v(self.xT)
            kc = (2*xi0**2*dv)**-0.5 * ((1 + 4*xi0**2*dv*self.kc**2)**0.5 - 1)**0.5
            return kc


class PotentialPoly(Potential):
    """ This class provides an implementation of a polynomial potential """

    def __init__(self, a=1.5, b=0.5):
        super().__init__()
        self.a = a
        self.b = b
        self.xF = 0.5 * (b - (b**2 + 4 * a**2)**0.5)
        self.xT = 0.5 * (b + (b**2 + 4 * a**2)**0.5)
        self.xminE = optimize.brentq(lambda x: self.v(x), self.xmax, self.xT)
        self.name = "poly-{}a-{}b".format(a, b)

    def v(self, x, xi=0):
        r""" Compute a scalar potential defined by

        .. math::
            v(x) = f(x) - (1 + \xi) f(x_T), \\
            f(x) = \frac{x^4}{4} - \frac{bx^3}{3} - \frac{a^2 x^2}{2}
        """

        def f(x):
            """ Lambda function """
            return x**4 / 4 - self.b * x**3 / 3 - self.a**2 * x**2 / 2

        return f(x) - (1 - xi) * f(self.xF) - xi * f(self.xT)

    def dv(self, x):
        r""" Compute the derivative of the scalar potential.

        .. math::
            v' = x(x^2 - bx - a^2)

        :param x: Scalar field
        :type x: float
        :rtype: float
        """
        return x * (x**2 - self.b * x - self.a**2)

    def ddv(self, x):
        r""" Compute the second derivative of the scalar potential.

        .. math::
            v'' = 3 x^2 - 2 b x - a^2.

        :param x: Scalar field
        :type x: float
        :rtype: float
        """
        return 3 * x**2 - 2 * self.b * x - self.a**2

class PotentialPolyNorm(Potential):
    """ This class provides an implementation of a normalized polynomial potential """

    def __init__(self, Dv):
        assert(Dv < 16/3)
        assert(Dv >= 0)

        super().__init__()
        self.Dv = Dv
        self.xF = -1
        self.xT = 1
        self.xminE = 1 - Dv/8 - (Dv*(16+Dv))**0.5/8
        self.kcmax = 10 if Dv > 0.8 else 0.8
        self.name = "poly-{}Dv".format(Dv)

    def v(self, x, xi=0):
        r""" Compute a scalar potential defined by

        .. math::
            v(x) = (1-x^2)^2 - 3/2 dv (x - x^3/3) - (1-2 \xi) dv
        """

        return (1-x**2)**2 - 3/4*self.Dv*(x-x**3/3) - (1-2*xi)/2*self.Dv

    def dv(self, x):
        r""" Compute the derivative of the scalar potential.

        .. math::
            v' = 4 (x + 3/8 dv) (x^2 - 1)

        :param x: Scalar field
        :type x: float
        :rtype: float
        """
        return 4*(x+3/16*self.Dv)*(x**2 - 1)

    def ddv(self, x):
        r""" Compute the second derivative of the scalar potential.

        .. math::
            v'' = 12 x^2 + 3 dv x - 4

        :param x: Scalar field
        :type x: float
        :rtype: float
        """
        return 12*x**2 + 3*self.Dv*x/2 - 4


class PotentialGauss(Potential):
    def __init__(self, dx, wa, wb, hb):
        """ Potential with two Gaussian whose maxima are separated by dx.
        Their width are wa and wb, and the ratio of height is hb. """
        super().__init__()
        self.wa = wa
        self.wb = wb
        self.hb = hb
        self.dx = dx
        self.xT = optimize.newton(self.dv, dx)-1e-14
        self.xF = optimize.newton(self.dv, 0)+1e-14
        self.xmax = optimize.brentq(self.dv, self.xF + 1e-4, self.xT - 1e-4)
        self.xminE = optimize.brentq(lambda x: self.v(x), self.xmax, self.xT)
        self.name = "gauss-{}-{}-{}-{}".format(dx, wa, wb, hb)

    def v(self, x, xi=0):
        def f(x):
            """ Lambda function """
            return -np.exp(-0.5*x**2/self.wa**2)\
                - self.hb*np.exp(-0.5*(x-self.dx)**2/self.wb**2)

        return f(x) - (1 - xi) * f(self.xF) - xi * f(self.xT)

    def dv(self, x):
        return x/self.wa**2 * np.exp(-0.5*x**2/self.wa**2) +\
            self.hb*(x-self.dx)/self.wb**2 *\
            np.exp(-0.5*(x-self.dx)**2/self.wb**2)

    def ddv(self, x):
        return -(x**2-self.wa**2)/self.wa**4 * np.exp(-0.5*x**2/self.wa**2) -\
            self.hb*((x-self.dx)**2-self.wb**2)/self.wb**4 *\
            np.exp(-0.5*(x-self.dx)**2/self.wb**2)


if __name__ == "__main__":
    pass
