#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r""" Provides a solver for the equations of motion """

# Dependencies
import numpy as np
from scipy import integrate as integrate
from scipy import optimize as optimize
import warnings
from . import potential
from . import misc

__author__ = "Maxime Trépanier"
__date__ = "01/30/2017"


# Generic class for the instanton configuration
class InstantonSolver:
    r""" This class defines a general integrator for the equations of motion.
    """
    def __init__(self, v, log=misc.Log()):
        """ The class requires a potential object on which to integrate. """
        assert(isinstance(v, potential.Potential))
        self.v = v
        self.log = log

    def edo(self, x, X):
        """ Compute the derivative of the system of equations to solve.
        The coordinate x is the field value, while X is the vector. """
        pass

    def setParameters(self):
        """ Set the free parameters for the a particular solution. """
        pass

    def solve(self):
        """ Returns a complete solution of the equations of motion. """
        pass


# Integrator for the EOM with GR
class InstantonSolverGR(InstantonSolver):
    r""" This class implements the system of equations with GR """
    def __init__(self, v, log=misc.Log(), d=4):
        super().__init__(v, log)
        self.d = d
        self.lDeltaF = 0
        self.lDeltaT = 0
        self.lDeltaFt0 = 0
        self.lDeltaTt0 = 0

    def edo(self, x, X):
        r""" The equations to solve are

        .. math::
            \frac{\partial r}{\partial x} = \sqrt{\frac{1+\kappa^2 r^2 \left(E-V_F^*\right)}{2 \left[ V+E-V_F^* \right]}},
            \frac{\partial E}{\partial x} = \frac{d-1}{r} \sqrt{\left[1+\kappa^2 r^2 \left(E-V_F^*\right) \right] 2 \left[ V+E-V_F^* \right]}.

        :param x: Scalar field
        :type x: float
        :param X: Solution vector
        :type X: [r, E]
        :rtype: dX
        """
        r, E = X
        A = 1+r**2*E/self.v.v(self.v.xF, self.xi)
        dphi = 2*(self.v.v(x, self.xi) + E)

        # If A cross zero, we reverse the sign of the equations
        if A < 1e-8 and not self.reverse:
            self.reverse = True

        # If dphi tends to zero, we are approaching the end point
        if dphi < 1e-10 and x < self.v.xmax:
            self.stop = True
            self.xstop = x

        # Another thing that can happen is if the instanton overshoots:
        # then E goes near vF but then diverges to infinity.
        if E > self.Emax and x < self.v.xmax:
            self.stop = True
            self.xstop = x
            self.overshoot = True

        if (dphi > 0):
            A = np.absolute(A)

            dr = 1/self.rdS*(A/dphi)**0.5
            dE = (r*self.rdS)**-1*(self.d-1)*(A*dphi)**0.5

            if self.reverse:
                dE = -dE
            else:
                dr = -dr
        else:
            dr = 0
            dE = 0

        dX = np.array([dr, dE])
        return dX

    def integrateBoundaryConditionsF(self, xf, Xf):
        """ This function finds the initial point xF* such that the equation
        for r is continuous. By matching the analytical solution for E with
        the numerical one, the slope of the solution should be continuous.

        :param xf: End coordinate of numerical integration
        :type xf: float
        :param Xf: End point of the solution
        :type Xf: [r: float, E: float]
        :rtype: [x, sol]
        """

        nbPtsx = 20
        if xf > self.v.xmax:
            raise ValueError("The stop point must be after the maximum")
        #  return Xf?

        def solSub(xFs, xf, lDelta=0, opt=False, rt0=False):
            if xFs > xf and opt:
                return 1

            vF = self.v.v(xFs, self.xi)
            l = self.v.dv(xFs)
            mu2 = self.v.ddv(xFs)
            if l < 0:
                l = 0
            t0 = 2*self.k*vF**0.5/(np.absolute(mu2) +
                                   2*(self.d-1)*self.k**2*vF)**0.5

            if rt0:
                return t0

            if lDelta == 0:
                Delta = (l/(np.absolute(mu2)/2 + (self.d-1)*self.k**2*vF))**0.5
            else:
                Delta = np.exp(lDelta)

            x = np.linspace(xFs+1e-14, xf, nbPtsx)
            E = -vF + (self.d-1)*self.k**2*vF*(x-xFs)**2

            if lDelta == 0:
                y = (x - xFs)**0.5/Delta
                if mu2 >= 0:
                    r = np.cos(t0*np.arcsinh(y))
                else:
                    r = np.cos(t0*np.arcsinh(y))
            else:
                r = np.cos(t0*(np.log(x - xFs)/2 - lDelta))
            if opt:
                return r[-1]
            else:
                r[0] = 0
                r[1] = 1
                return x, np.array([r, E]).T

        def solSup(xFs, xf, lDelta=0, opt=False, rt0=False):
            vF = self.v.v(xFs, self.xi)
            l = self.v.dv(xFs)
            mu2 = self.v.ddv(xFs)
            if l < 0:
                l = 0
            t0 = 2*self.k*vF**0.5*(self.d+2)**0.5 /\
                (3*np.absolute(mu2)+2*(self.d-1)*self.k**2*vF)**0.5

            if rt0:
                return t0

            if lDelta == 0:
                Delta = ((self.d+2)/self.d*l / (3*np.absolute(mu2)/2 +
                                                (self.d-1)*self.k**2*vF))**0.5
            else:
                Delta = np.exp(lDelta)

            x = np.linspace(xFs+1e-14, xf, nbPtsx)
            E = -vF - (self.d-1)/self.d*l*(x - xFs) -\
                (self.d-1)/(self.d+2)*(mu2/2 - self.k**2*vF)*(x-xFs)**2
            if lDelta == 0:
                y = (x - xFs)**0.5/Delta
                r = np.sin(t0*np.arcsinh(y))
                if opt and t0*np.arcsinh(y[-1]) > np.pi/2:
                    return Xf[0] + 1e-1
            else:
                r = np.sin(t0*(np.log(x - xFs)/2 - lDelta))
                if opt and t0*(np.log(x[-1] - xFs)/2 - lDelta) > np.pi/2:
                    return Xf[0] + 1e-1
            if opt:
                return r[-1]
            else:
                r[0] = 0
                return x, np.array([r, E]).T

        lDelta = 0
        self.lDeltaF = 0
        if not self.reverse:  # Subcritical case
            try:
                xFs = optimize.brentq(lambda x: solSub(x, xf, opt=True)-Xf[0],
                                      self.v.xF+1e-8, xf-1e-12)
            except ValueError:
                xFs = self.v.xF
                t0 = solSub(xFs, xf, rt0=True)
                lDelta = 0.5*np.log(xf-xFs) - (np.arccos(Xf[0]))/t0
                self.lDeltaF = 0.5*np.log(xf-xFs) - (np.pi-np.arcsin(Xf[0]))/t0
            return solSub(xFs, xf, lDelta=lDelta)
        else:  # Supercritical case
            if Xf[1] + self.v.v(xf, self.xi) <= 1e-3 and Xf[0] < 1e-3:  # Already a solution
                vF = self.v.v(xf, self.xi)
                r = np.array([0,0])
                E = np.array([-vF, -vF])
                return np.array([xf, xf]), np.array([r, E]).T

            try:
                xFs = optimize.brentq(lambda x: solSup(x, xf, opt=True)-Xf[0],
                                      self.v.xF+1e-12, xf-1e-12)
            except ValueError:
                if xf-self.v.xF > 5e-1:
                    raise ValueError("Can't find xFs")
                xFs = self.v.xF
                t0 = solSup(xFs, xf, rt0=True)
                theta = np.arcsin(Xf[0])
                lDelta = 0.5*np.log(xf-xFs) - theta/t0
                self.lDeltaF = lDelta
            return solSup(xFs, xf, lDelta=lDelta)

    def integrateBoundaryConditionsT(self, xTs, lDelta=0):
        """ This function integrates analytically the EOM near xT* away from the
        singularity. If lDelta is not zero, than the function computes the
        solution for a xFs asymptotically close to xF. """

        # Function parameters
        maxv0 = 1e-6  # Stop integration at when v0 reaches maxv0. Control xmin
        nbPtsx = 20  # Number of points in the solution

        # Shortcuts
        vF = self.v.v(self.v.xF, self.xi)
        vT = np.absolute(self.v.v(xTs, self.xi))
        mu2n = False

        # Approximation of the potential
        l = self.v.dv(xTs)
        mu2 = self.v.ddv(xTs)
        if mu2 < 0:
            mu2n = True
            mu2 = -mu2
        t0 = 2*self.k*vT**0.5*(self.d+2)**0.5 /\
            (3*mu2+2*(self.d-1)*self.k**2*vT)**0.5
        if lDelta == 0:
            Delta = np.absolute((self.d+2)/self.d*l /
                                (3*mu2/2 + (self.d-1)*self.k**2*vT))**0.5
        else:
            Delta = np.exp(lDelta)

        # v0 function to determine xmin
        def v0(x):
            return l/self.d*x+(3*mu2/2+(self.d-1)*self.k**2*vT)/(self.d+2)*x**2
        try:
            xmin = optimize.brentq(lambda x: v0(x)-maxv0, 0, 1e-2)
        except:
            xmin = 1e-1

        # Compute the solution
        x = np.linspace(xTs, xTs - xmin, nbPtsx)
        E = -self.v.v(xTs, self.xi) - (self.d-1)/self.d*l*(x-xTs) -\
            (self.d-1)/(self.d+2)*(mu2/2-self.k**2*vT)*(x-xTs)**2

        if lDelta == 0:
            y = (xTs-x)**0.5/Delta
            if mu2n:
                r = np.sinh(t0*np.arcsinh(y))
            else:
                r = np.sinh(t0*np.arcsinh(y))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = np.sinh(t0*(0.5*np.log(xTs - x) - lDelta))
            r[0] = 0
        r = (vF/vT)**0.5*r

        # Save lDelta for log
        self.lDeltaT = lDelta
        self.lDeltaTt0 = lDelta*t0

        return x, np.array([r, E]).T

    def R(self):
        """ This function computes the action ratio R for the previously found
        solution. """
        xmin = np.where(self.x > self.xstop)[0]
        if len(xmin) == 0:
            return 0
        else:
            xmin = xmin[0]
        r, E = self.sol.T
        v0 = np.absolute(self.v.v(self.x, self.xi) + E)
        vF = self.v.v(self.v.xF, self.xi)
        vFs = self.v.v(self.x[0], self.xi)
        rmax = np.argmax(r)
        rbar = (vFs/vF)**0.5*r[xmin]

        # We complete the contour of the integration
        Ra = vF/vFs*(1-(1-rbar**2)**1.5)/2
        if rmax > xmin: # Numerical integration covers the peak rmax -> take the absolute value
            Rn1 = 3/4*integrate.simps((1+r[xmin:rmax]**2*E[xmin:rmax]/vF)**0.5, r[xmin:rmax]**2)
            Rn2 = -3/4*integrate.simps((1+r[rmax:]**2*E[rmax:]/vF)**0.5, r[rmax:]**2)
            R1 = Rn1 + Ra
        else:
            R1 = Ra
            Rn2 = -3/4*integrate.simps((1+r[xmin:]**2*E[xmin:]/vF)**0.5, r[xmin:]**2)
        # We must reverse the sign of R1 for subcritical instantons
        if round(self.k/self.kc, 8) < 1:
            R1 = 1-R1

        Rn3 = -3/4*self.k/vF**0.5*integrate.trapz(r**3*(2*v0)**0.5, self.x)
        return R1 + Rn2 + Rn3

    def resetFlags(self):
        self.reverse = False
        self.stop = False
        self.overshoot = False
        self.xstop = self.v.xF
        self.Emax = 0

    def setParameters(self, k, xi):
        self.k = k
        self.xi = xi
        self.rdS = 1/(k*self.v.v(self.v.xF, xi)**0.5)
        self.x = None
        self.sol = None
        self.kc = self.v.kstar(xi)
        self.resetFlags()

    def shoot(self, x0, lDelta=0, rsol=False):
        """ This function integrate the EOM from a given x0, until x is
        close to xF*, where it stops and returns the numerical solution
        (if rsol = True) or the angular difference of the r solution.

        If lDelta is not 0, then x0 is assumed to be asymptotically close
        to xT. """

        # We start by resetting the integration flags
        self.resetFlags()

        # We integrate analytically the singularity
        if lDelta == 0:
            xa0, Xa0 = self.integrateBoundaryConditionsT(x0)
        else:
            xa0, Xa0 = self.integrateBoundaryConditionsT(x0, lDelta=lDelta)
        self.Emax = -self.v.v(x0, self.xi)

        # Numerical integration
        integrator = integrate.ode(self.edo).set_integrator('dopri5', nsteps=2000)
        integrator.set_initial_value(Xa0[-1], xa0[-1])
        # If the flag stop is active, we stop the integration
        integrator.set_solout(lambda x, X: -1 if self.stop else 0)

        if xa0[-1] == self.v.xmax:
            xnum = np.linspace(self.v.xmax, self.v.xF, 500)[1:]
        else:
            xnum = np.concatenate([np.linspace(xa0[-1], self.v.xmax, 100)[1:],
                                   np.linspace(self.v.xmax, self.v.xF, 500)[1:]])
        Xnum = []
        for xi in xnum:
            integrator.integrate(xi)
            if not integrator.successful():
                Xnum.append(Xa0[-1])
                print("k: {}, x: {}: Unexpected error during integration.".format(self.k, x0))
                break
            Xnum.append(integrator.y)

        # We look for the last valid point of the integration
        xmax = np.where(xnum > self.xstop)[0]
        if len(xmax) != 0:
            xmax = xmax[-1]
            if xmax == 0:
                xmax = 1
            xnum = xnum[:xmax]
            Xnum = np.array(Xnum[:xmax])
        else:
            xnum = np.array([xa0[-1]])
            Xnum = np.array([Xa0[-1]])

        # If the integrator didn't stop until xF but r != 0, then we know it
        # overshoots
        if not self.stop and Xnum[-1][0] != 0:
            self.overshoot = True

        x = np.concatenate([xa0, xnum])
        X = np.concatenate([Xa0, Xnum])

        self.xstop = x[-1]

        # If the radius of the bubble exceeds that of dS space, then it can't
        # in the space, so the solution is invalid
        if len(np.where(np.nan_to_num(X.T[0]) >= 1)[0]) != 0:
            self.overshoot = True
            if rsol:
                raise ValueError("The solution did not converge")
            else:
                return 1

        # We compute the analytical solution at the end
        try:
            xaf, Xaf = self.integrateBoundaryConditionsF(xnum[-1], Xnum[-1])
        except ValueError:
            if rsol:
                raise ValueError("The solution did not converge")
            else:
                return -1

        # If we return the solution
        if rsol:
            x = np.concatenate([x, xaf[::-1]])
            X = np.concatenate([X, Xaf[::-1]])
            r, E = X.T
            E -= E[-1]
            X = np.array([r, E]).T
            return x, X
        # If not, we compute the slope at the end of the numerical integration
        else:
            # If there is a single point, then
            if len(xaf) == 1 and Xaf[0][0] != 0:
                return -1
            return Xnum[-1][1] - Xaf[-1][1]
            # drnum = self.edo(xnum[-1], Xnum[-1])[0]
            # dra = (Xaf[-1][0] - Xaf[-2][0])/(xaf[-1] - xaf[-2])
            # deltadr = np.absolute(np.arctan(drnum) - np.arctan(dra))
            # # We add an extra factor if it undershoots because we don't want
            # # the solution
            # return deltadr if self.overshoot else -deltadr

    def solve(self, verbose=False):
        """ This function finds the optimal x0 such that the boundary
        conditions are satisfied. It returns True if succeeded. """

        k = round(self.k/self.kc, 8)
        xi = round(self.xi, 8)
        if (k, xi) in self.log.data:
            x0, lDelta = self.log.data[(k, xi)]
            x, sol = self.shoot(x0, lDelta=lDelta, rsol=True)
            self.x = x[::-1]
            self.sol = sol[::-1]
            return True

        # The goal here is to find the appropriate x0 such that the equations
        # of motion satisfy the boundary conditions. We use a shooting method
        # to test each solution and explore the parameter space.
        xmin = self.v.xminE
        xmax = self.v.xT-1e-8
        x0 = 0

        # We try to find a solution in the normal range
        thinwall = False
        thermal = False
        try:
            x0 = optimize.brentq(self.shoot, xmin, xmax, rtol=1e-12)
        # If it fails, then either we are in the thin-wall regime and the field
        # starts asymptotically close to xT, or we are in the thermal regime
        # and the instanton starts and ends far from both endpoints.
        except ValueError as err:
            if self.overshoot:
                thermal = True
            else:
                thinwall = True

        lDelta = 0
        if thermal:
            try:
                xmin = self.v.xmax+1e-2
                xmax = self.v.xminE
                x0 = optimize.brentq(self.shoot, xmin, xmax, rtol=1e-10)
            except ValueError:
                return False
        if thinwall:
            try:
                x0 = self.v.xT
                lDelta = optimize.brentq(lambda x: self.shoot(x0, lDelta=x), -8, -50)
            except ValueError:
                return False

        try:
            x, sol = self.shoot(x0, lDelta=lDelta, rsol=True)
        except ValueError as err:
            if verbose:
                print(err)
            return False
        self.log.write(round(self.k/self.kc, 8), self.xi, x0, lDelta)
        self.x = x[::-1]
        self.sol = sol[::-1]

        return True

    def thinwallParameters(self):
        xF = self.x[0]
        xT = self.x[-1]
        v0 = self.sol.T[1] + self.v.v(self.x, xi=0)
        Sfs = integrate.trapz(np.absolute(2*v0)**0.5, self.x)
        vFs = self.v.v(xF, self.xi)
        vTs = self.v.v(xT, self.xi)
        # Dv = self.v.v(self.v.xF, self.xi) - self.v.v(self.v.xT, self.xi)
        Dvs = vFs-vTs

        if Dvs < 0:
            raise ValueError("Invalid transition")

        ksm = (self.d-2)/(self.d-1)*Dvs**0.5/Sfs
        alpha = (self.k/ksm)**2
        xis = vFs/Dvs

        return alpha, xis
