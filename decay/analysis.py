#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r""" Provides tools to analyze a solution of the equation of motion """

import numpy as np


def thinwallSol(alpha, xi, EED=1):
    """ This function computes the parameters of the thin-wall solution,
    the radius of the bubble and the action ratio R. The EED corresponds
    to the effective energy density and is vF/vFs. """

    rbar = EED**0.5*(4*alpha*xi/((alpha-1)**2+4*alpha*xi))**0.5
    R = EED*(1 - 0.5*(1-xi)**-1*(1 + (alpha*(1-2*xi)-1)/((alpha-1)**2+4*alpha*xi)**0.5))

    return rbar, R


def thinWallSol2(v, alpha, xi, xFs, xTs, lDeltaF=0, lDeltaT=0):
    """ This function returns an estimate of r and E in the thin-wall
    approximation. """

    x = np.linspace(v.xF+1e-14, v.xT, 100)
    E = 1/2*(v.v(x) - v.v(v.xF+v.xT-x) + v.v(v.xF) + v.v(v.xT))

    # r near xFs
    if lDeltaF != 0:
        xFs = v.xF
    l = v.dv(xFs)
    mu2 = v.ddv(xFs)
    if mu2 == 0:
        mu2 = 1e-15
    mu = np.absolute(mu2)**0.5
    beta = 2*alpha**0.5*v.kstar()*v.v(xFs, xi)**0.5/mu
    if lDeltaF == 0:
        Delta = (2*l)**0.5/mu
    else:
        Delta = np.exp(lDeltaF)

    if lDeltaF == 0:
        rFs = np.sin(beta*np.arcsinh((x-xFs)**0.5/Delta))
    else:
        rFs = np.sin(beta*(0.5*np.log(x-xFs) - lDeltaF))
        rFs[0] = 0
        if alpha > 1:
            rFs[1] = 1

    # r near xTs
    if lDeltaT != 0:
        xTs = v.xT
    l = v.dv(xTs)
    mu2 = v.ddv(xTs)
    if mu2 == 0:
        mu2 = 1e-15
    mu = np.absolute(mu2)**0.5
    beta = 2*alpha**0.5*v.kstar()*v.v(xFs, xi)**0.5/mu
    if lDeltaT == 0:
        Delta = (-2*l)**0.5/mu
    else:
        Delta = np.exp(lDeltaT)
    #print(xTs, l, mu)

    if lDeltaT == 0:
        y = (xTs-x)**0.5/Delta
        if mu2 > 0:
            rTs = np.sinh(-beta*np.arcsinh(y))
        elif mu2 < 0:
            rTs = np.sinh(-beta*np.arcsin(y))
    else:
        rTs = np.sinh(-beta*(0.5*np.log(x-xTs) - lDeltaT))
        rTs[0] = 0

    #print(rFs, rTs)
    r = np.concatenate([rFs[0:50], rTs[50:]])
    r = rFs
    return x, E, r
