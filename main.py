#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r""" Data generation script for vacuum decay project.

This script uses the decay library to calculate instanton solutions in
semiclassical GR for 3 choices of potentials with different depth. The data
generated is analysed in the thesis and compared to the thin-wall approximation.
"""

# Dependencies
from decay import *
import numpy as np
from matplotlib import pyplot as plt
import csv

__author__ = "Maxime Trépanier"
__date__ = "01/30/2017"
# note: comments added on 29/3/2024

def write_data_dict(filename, d):
    """ Writes the dictionary d to the location "filename".
    """
    with open(filename, "w") as f:
        writer = csv.writer(f, delimiter=" ")  # opens file
        writer.writerow(d.keys())  # writes header
        for vi in zip(*d.values()):
            writer.writerow(vi)  # writes values

def write_solution(filename, gr):
    """ Writes a solution of InstantonSolverGR to the file filename.
    """
    with open(filename, "w") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerow(["x", "r", "E"])
        for xi, soli in zip(gr.x, gr.sol):
            writer.writerow([xi, soli[0], soli[1]])


if __name__ == "__main__":
    # Compare three potentials with different depth
    v1 = potential.PotentialPolyNorm(0.5)  # Mid
    v2 = potential.PotentialPolyNorm(1)  # Deep potential
    v3 = potential.PotentialPolyNorm(0.2)  # Shallow potential (thin-wall approximation)

    # EXAMPLES OF SOLUTIONS

    # SUBCRITICAL, CRITICAL AND SUPERCRITICAL SOLNS:
    # Instantons are characterised by two parameters alpha and xi
    # (potential, alpha, xi)
    to_solve = [(v1, 0.5, 0.1), (v1, 1, 0.1), (v2, 1.5, 0.1)]
    for v, a, xi in to_solve:
        log = misc.Log("Logs/{}".format(v.name))  # log for debugging
        gr = solver.InstantonSolverGR(v, log)  # solver with potential v
        kc = v.kstar(xi)  # get critical value of k
        gr.setParameters(a*kc, xi)  # k is alpha*kc
        if gr.solve():  # if successful
            # save soln
            write_solution("Data/{}-{}a-{}x.dat".format(v.name, a, xi),
                           gr)

    # SMALL ALPHA, XI:
    to_solveSG = [(v1, 1e-3, 1e-2), (v1, 1e-2, 1e-2), (v1, 1e-1, 1e-2)]
    for v, a, xi in to_solveSG:
        log = misc.Log("Logs/{}".format(v.name))
        gr = solver.InstantonSolverGR(v, log)
        kc = v.kstar(xi)
        gr.setParameters(a*kc, xi)
        if gr.solve():
            # calculates dE from the solution
            x = gr.x
            E = gr.sol.T[1]
            E = E-E[0]
            Eth = 3/4*v.Dv*(x-x**3/3)
            Eth = Eth-Eth[0]
            dE = (Eth - E)/v.Dv
            d = {'x': gr.x, 'E': E, 'dE': dE}  # dictionary for saving
            write_data_dict("Data/{}-{}a-{}x.dat".format(v.name, a, xi), d)

    # SMALL XI, ARBITRARY ALPHA:
    to_solveTW = [(vi, ki, 0.1) for ki in [0.2, 0.6, 1, 1.4, 2.2] for vi in [v1, v2, v3]]
    for v, a, xi in to_solveTW:
        log = misc.Log("Logs/{}".format(v.name))
        gr = solver.InstantonSolverGR(v, log)
        kc = v.kstar(xi)
        gr.setParameters(a*kc, xi)
        if gr.solve():
            # calculates rb, the thin-wall solution to r
            alpha, xis = gr.thinwallParameters()
            EED = v.v(v.xF, gr.xi)/v.v(gr.x[0], gr.xi)
            rbar, R = analysis.thinwallSol(alpha, xis, EED)
            rthin = rbar*np.ones(len(gr.x))
            rthin[0] = 0
            rthin[-1] = 0
            d = {'x': gr.x, 'r': gr.sol.T[0], 'E': gr.sol.T[1], 'rb': rthin}
            write_data_dict("Data/{}-{}a-{}x.dat".format(v.name, a, xi), d)

    # MODULI SPACE OF INSTANTONS (fixed potential, xi, range of alpha)
    to_solveEP = [(v1, 1e-1, 0.1, 3.6, 0.1), (v2, 1e-1, 0.1, 2.6, 0.1), (v3, 1e-1, 0.1, 3.4, 0.1)]
    for v, xi, kmin, kmax, kstep in to_solveEP:
        log = misc.Log("Logs/{}".format(v.name))
        gr = solver.InstantonSolverGR(v, log)
        kc = v.kstar(xi)

        k = np.arange(kmin, kmax, kstep)*kc
        xFs = []  # starting point of soln
        xTs = []  # end point of soln
        lDeltaF = []  # log of Delta_F
        lDeltaT = []  # log of Delta_T
        R = []  # transition amplitude
        twp = []  # thin-wall parameters
        EED = []
        for ki in k:
            # for each ki, add soln to arrays
            gr.setParameters(ki, xi)
            if gr.solve():
                xTs.append(gr.x[-1])
                xFs.append(gr.x[0])
                if gr.lDeltaF == 0:
                    lDeltaF.append(np.log(gr.x[0] - gr.v.xF))
                else:
                    lDeltaF.append(2*gr.lDeltaF)
                if gr.lDeltaT == 0:
                    lDeltaT.append(np.log(gr.v.xT-gr.x[-1]))
                else:
                    lDeltaT.append(2*gr.lDeltaT)
                R.append(gr.R())
                twp.append(gr.thinwallParameters())
                EED.append(gr.v.v(gr.v.xF, xi)/gr.v.v(gr.x[0], xi))
            else:
                # solution failed
                xTs.append(np.nan)
                xFs.append(np.nan)
                lDeltaT.append(np.nan)
                lDeltaF.append(np.nan)
                R.append(np.nan)
                twp.append([np.nan, np.nan])
                EED.append(np.nan)
                continue

        # convert arrays to numpy
        xTs = np.array(xTs)
        xFs = np.array(xFs)
        lDeltaF = np.array(lDeltaF)
        lDeltaT = np.array(lDeltaT)
        R = np.array(R)
        twp = np.array(twp)
        EED = np.array(EED)
        alpha, xis = twp.T
        # dictionary for saving
        d = {'k': k/kc, 'xFs': xFs, 'xTs': xTs, 'lDeltaF': lDeltaF,
             'lDeltaT': lDeltaT, 'R': R, 'alpha': alpha, 'xis': xis}
        # save
        write_data_dict("Data/{}-{}x-EP.dat".format(v.name, xi), d)
