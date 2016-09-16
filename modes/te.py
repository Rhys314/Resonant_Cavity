# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:37:29 2016

@author: rpoolman
"""
import numpy as np
from .utilities import _grad2D


class TERectangularBox:
    """
    A functor that calculates transverse electric modes in a resonant cavity
    of rectangualr cross section.  See Jackson section 8.4, pp. 361 and section
    8.7, pp.368.
    """
    def __init__(self, a, b, c, m, n, p, mu, epsilon, f, H0=1):
        """
        Parameters
        ----------
        a: double
            The size of the resonant cavity along the x direction.

        b: double
            The size of the resonant cavity along the y direction.

        c: double
            The size of the resonant cavity along the z direction.

        m: int
            The number of half wavelengths along the x direciton.

        n: int
            The number of half wavelengths along the y direciton.

        p: int
            The number of half wavelengths along the z direciton.

        mu: double
            The permeability of the cavity.

        epsilon: double
            The permittivity of the cavity.

        f: double
            The frequency of the wave.

        H0: double
            The initial amplitude of the wave, defaults to unity.
        """
        self.H0 = H0
        self.a = a
        self.b = b
        self.c = c
        self.m = m
        self.n = n
        self.m = m
        self.p = p
        self.mu = mu
        self.epsilon = epsilon
        self.omega = 2*np.pi*f

    def __call__(self, x, y, z):
        unit_z = np.zeros(3)
        unit_z[2] = 1
        Hz = self._psi(x, y)*np.sin(self.p*np.pi*z/self.c)
        grad_psi = np.append( _grad2D(self._psi, np.array([x, y]), 0.1),
            0.0)
        Et = -1j*self.omega*self.mu/self._gamma() * \
            np.sin(self.p*np.pi*z/self.c)*np.cross(unit_z, grad_psi)
        Ht = self.p*np.pi/(self.c*self._gamma()**2)*grad_psi
        return Et, Ht, Hz

    def _psi(self, x, y):
        """
        The transverse components of the wave function.

        Parameters
        ----------
        x: double
            A x coordinate.

        y: double
            A y coordinate

        Returns
        -------
        psi: double
            The value of the transverse wavefunction at (x, y)
        """
        return self.H0*np.cos(self.m*np.pi*x/self.a) * \
            np.cos(self.n*np.pi*y/self.b)

    def _gamma(self):
        """
        Characterises the mode.

        Returns
        -------
        gamma: double
            The mode characteristic.
        """
        return np.lib.scimath.sqrt(self.mu*self.epsilon*self.omega**2 -
                                   (self.p*np.pi/self.c)**2)
