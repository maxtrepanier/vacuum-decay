# Vaccum decay

## Introduction

This project is part of my master's thesis and the results of my paper [^1]. It
It studies the problem of vacuum decay, a quantum tunneling event in
semiclassical general relativity (GR) between a metastable (_false_) vacuum and a
stable (_true_) vacuum, driven by a scalar field.

The code attached here aims at solving a set of ODEs appearing in that context,
and analysing the properties of their solutions.
Given a potential $v(x)$ with two minima, it calculates the motion of the field
$x$ in that potential, subject to the differential equations
$$
\frac{d^2x}{ds^2} + \frac{3}{r}\frac{dr}{ds}\frac{dx}{ds}
=
\frac{dv}{dx},
$$
$$
\left(\frac{dr}{ds}\right)^2
=
1+\kappa^2r^2 \left(
\frac{1}{2} \left(\frac{dx}{ds}\right)^2
- v(x).
$$
$r(s)$ is an auxiliary field roughly interpreted as the curvature radius of the
universe.

The equations are parametrised by $\kappa$, which controls the strength of the
gravitational interaction, and $\xi$, which controls the energy density of the
false vacuum.

## The code

The repository contains the package decay, containing 4 files:
 - potential.py: Defines a class Potential implementing $v(x)$
 - solver.py: Defines a class InstantonSolver, defining the ODE and solving the
   boundary value problem.
 - analysis.py: Implements two auxiliary functions calculating the parameters of
   the analytic solution in the thin-wall approximation.
 - misc.py: Defines auxiliary methods to generate various plots and a class Log
   used for debugging.

In addition, I include the file main.py, which contains a script generating all
the data relevant for the project (examples of solutions for various choices of
potentials, calculation of the rate of tunneling).

[^1]: Espinosa, Fortin, Trépanier; Consistency of scalar potentials from quantum
  de Sitter space, [Phys. Rev. D 93, 124067](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.93.124067)
