# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:36:22 2016
The main script for using the resonant cavity module.
@author: rpoolman
"""
import numpy as np
import modes

# number of points
N = 10  # x direction
M = 10  # y direction
P = 10  # z direction

# defines the mode of interest
n = 1  # x direction
m = 0  # y direction
p = 1  # z direction

# defines the box size
a = 1  # x direction
b = 1  # y direction
c = 1  # z direction

# define box material via permitivity and permeability
mu = 1.0
epsilon = 1.0

# define frequency of wavelength
f = 1/10

# calculation
te = modes.TERectangularBox(a, b, c, m, n, p, mu, epsilon, f)
x_array = np.linspace(0, a, N)
y_array = np.linspace(0, b, M)
z_array = np.linspace(0, c, P)
E = np.zeros((N, M, P, 3), dtype=np.complex128)
H = np.zeros((N, M, P, 3), dtype=np.complex128)

for ii, x in enumerate(x_array):
    for jj, y in enumerate(y_array):
        for kk, z in enumerate(z_array):
            Et, Ht, Hz = te(x, y, z)
            E[ii, jj, kk] = np.array([Et[0], Et[1], 0])
            H[ii, jj, kk] = np.array([Ht[0], Ht[1], Hz])
            test = np.dot(E[ii, jj, kk], H[ii, jj, kk])
            if test != 0:
                print("E and H not orthognal.")
                print("E.H = {}".format(test))

modes.plot(x_array, y_array, z_array, E, H)