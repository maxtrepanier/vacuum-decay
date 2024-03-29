#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r""" Miscellaneous functions """

# Dependencies
import numpy as np
from matplotlib import pyplot as plt
import math
from . import analysis


def round_sig(x, sig=1):
    """ Round up to a sig significative digits """
    if x == 0 or x == np.inf:
        return x
    return round(x, sig-int(math.floor(math.log10(abs(x))))-1)


def plot(gr):
    x = gr.x
    r, E = gr.sol.T
    v = gr.v

    alpha, xi = gr.thinwallParameters()
    EED = v.v(v.xF, gr.xi)/v.v(x[0], gr.xi)
    rbar, R = analysis.thinwallSol(alpha, xi, EED)
    rthin = rbar*np.ones(len(gr.x))
    rthin[0] = 0
    rthin[-1] = 0

    fig = plt.figure(0, figsize=(12, 4))
    ax = fig.add_subplot("121")
    ax.plot(x, r, 'b', lw=2)
    ax.plot(x, rthin, '--k')
    ax.set_ylabel(r"$r/\tilde{r}$", fontsize=18)
    ax.set_xlabel("$x$", fontsize=18)
    ax.set_xlim(x[0]-0.1, x[-1]+0.1)

    ax2 = fig.add_subplot("122")
    ax2.fill_between(x, 0, v.v(x)+E-v.v(x[0]), facecolor='b', alpha=0.2)
    ax2.plot(x, -E+v.v(x[0], gr.xi), 'b', lw=2)
    ax2.plot(x, v.v(x, gr.xi), 'k', lw=2)

    ymax = max(np.max(v.v(x)+E-v.v(x[0])), 1.1*v.v(v.xmax, gr.xi))

    # Background lines
    xv = np.linspace(v.xF-0.1, v.xT+0.1, 300)
    ax2.plot(xv, v.v(xv, gr.xi), 'k')
    ax2.set_ylabel(r"$v(x)$", fontsize=18)
    ax2.set_xlabel("$x$", fontsize=18)
    ax2.set_ylim(1.1*v.v(v.xT, gr.xi)-0.1, ymax+0.1)
    ax2.set_xlim(v.xF-0.1, v.xT+0.1)
    plt.tight_layout()
    plt.show()


def plotR(gr, k, xi):
    kc = gr.v.kstar(xi)
    # xTs = []
    # xFs = []
    lDeltaF = []
    lDeltaT = []
    R = []
    twp = []
    EED = []
    for ki in k:
        gr.setParameters(ki, xi)
        if gr.solve():
            # xTs.append(gr.x[-1])
            # xFs.append(gr.x[0])
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
            # xTs.append(np.nan)
            # xFs.append(np.nan)
            lDeltaT.append(np.nan)
            lDeltaF.append(np.nan)
            R.append(np.nan)
            twp.append([np.nan, np.nan])
            EED.append(np.nan)
            continue
    # xTs = np.array(xTs)
    # xFs = np.array(xFs)
    lDeltaF = np.array(lDeltaF)
    lDeltaT = np.array(lDeltaT)
    R = np.array(R)
    twp = np.array(twp)
    EED = np.array(EED)
    alpha, xis = twp.T

    fig = plt.figure(0, figsize=(12, 4))
    ax = fig.add_subplot("121")
    ax.plot(k/kc, np.log(-lDeltaF), 'b', lw=2)
    ax.plot(k/kc, np.log(-lDeltaT), 'b', lw=2)
    # ax.set_ylim(max(3*np.min(lDeltaT), np.min(lDeltaF)), 0)
    ax2 = fig.add_subplot("122")
    ax2.plot(k/kc, R, 'bo')
    ax2.plot(k/kc, analysis.thinwallSol(alpha, xis, EED)[1], lw=2)
    ax2.plot(k/kc, analysis.thinwallSol((k/kc)**2, xi)[1], 'b--')
    ax2.set_ylim(0, 1)
    plt.show()


class Log():
    def __init__(self, fname=""):
        self.data = dict()
        self.fname = fname
        self.read()

    def write(self, k, xi, x0, lDelta):
        self.data[(k, xi)] = (x0, lDelta)

    def read(self):
        if self.fname == "":
            return
        try:
            with open(self.fname, 'r') as f:
                for line in f.readlines():
                    k, xi, x0, lDelta = line.split(' ')
                    self.data[(float(k), float(xi))] = (float(x0), float(lDelta))
        except FileNotFoundError:
            pass

    def save(self):
        if self.fname == "":
            return
        with open(self.fname, 'w') as f:
            for k, xi in sorted(self.data):
                f.write("{:.8f} {:.8f} {} {}\n".format(k, xi,
                                                       self.data[(k, xi)][0],
                                                       self.data[(k, xi)][1]))
