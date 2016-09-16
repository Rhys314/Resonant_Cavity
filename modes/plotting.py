# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:42:39 2016
Plots the electric and magnetic fields using mayavi quiver plot.
@author: rpoolman
"""
import numpy as np
import mayavi.mlab as mlab


def plot(x_array, y_array, z_array, E, H):
    """
    Plots two mayavi scenes one with the electric field and one with magenetic.
    Uses the quiver3d plot to do so.

    Parameters
    ----------
    x-array: 1D double array
        The x-coordinates.

    y-array: 1D double array
        The y-coordinates.

    z-array: 1D double array
        The z-coordinates.

    E: 3D array of 3-vectors
        The electric field, each element in the array is an electric field
        vector E = (E_x, E_y, E_z)^T

    H: 3D array of 3-vectors.
        The magnetic field, each element in the array is an magnetic field
        vector H = (H_x, H_y, H_z)^T
    """
    X, Y, Z = np.meshgrid(x_array, y_array, z_array)
    mlab.figure('Real Electric Field')
    mlab.clf()
    mlab.quiver3d(X, Y, Z,
                  np.real(E[:, :, :, 0]), np.real(E[:, :, :, 1]),
                  np.real(E[:, :, :, 2]))
    mlab.figure('Real Magnetic Field')
    mlab.clf()
    mlab.quiver3d(X, Y, Z,
                  np.real(H[:, :, :, 0]), np.real(H[:, :, :, 1]),
                  np.real(H[:, :, :, 2]))
