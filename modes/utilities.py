# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:52:11 2016
Contains classes and functions that provided services for the main
calculations.
@author: rpoolman
"""
import numpy as np


def __doNothing(temp):
    return temp


def _grad2D(func, x, h, useRidders=False):
    """
    This function calculates the 2D Cartesian gradiant of a function at a
    point using the function derivative along each axis.  An estimate of the
    error in the derivative is returned.

    Parameters
    ----------
    func: function
        The function which is to have it's derivative calcualted.  Must take
        three arguments.
    x: 2 element double array
        The 2D Cartesian vector that defines the point at which the derivative
        is to be calculated.
    h: 2 element double array
        A first guess at the stepsize for the derivative.  A 2D Cartiesian
        vector.
    useRidders: bool
        Defaults to False and causes the symmetrical difference method to be
        used, if set to true then the symmetrical difference is used.  If the
        former is used then no error is returned.

    Returns
    -------
    gradf: 2 element double array
        The derivative grad f evaulated at the parameter x.
    error: double
        An estimate of the error between numerical derivative and the exact
        value.  If useRidders is set to False then no error is returned.
    """
    # check
    if len(x) != 2 and len(h) != 2:
        raise ValueError('grad2D: x and h should be 2D Cartesian vectors.')

    # arrays
    gradf = np.zeros(2)
    err = None
    if (useRidders):
        err = np.zeros(2)
        # sets maximu size of the tableau
        ntab = 10
        # stesize decreased by con at each iteration
        con = 1.4
        con2 = con*con
        numerical_limits = np.finfo('d')
        big = numerical_limits.max
        # return when error is SAFE worst than the best so far
        safe = 2.0
        a = np.zeros([ntab, ntab])

        if (h[0] == 0.0 or h[1] == 0.0):
            raise ValueError("grad: All stepsize h elements must be non-zero.")

        hh = h

        # x direction
        a[0, 0] = (func(x[0] + hh[0], x[1]) -
                   func(x[0] - hh[0], x[1]))/(2.0*hh[0])
        err[0] = big
        for ii in range(ntab):
            # successive columns in the Neville tableau will go to smaller
            # stepsizes and higher orders of extrapolation
            hh[0] = hh[0]/con
            # trying a new smaller stepsize
            a[0][ii] = (func(x[0] + hh[0], x[1]) -
                        func(x[0] - hh[0], x[1],))/(2.0*hh[0])
            fac = con2

            for jj in range(1, ii + 1):
                a[jj, ii] = (a[jj - 1, ii]*fac - a[jj - 1, ii - 1])/(fac - 1.0)
                fac = con2*fac
                errt = np.max([np.abs(a[jj, ii] - a[jj - 1, ii]),
                               np.abs(a[jj, ii] - a[jj - 1, ii - 1])])
                # the error strategy is to compare each new extrapolation to
                # one order lower both at the present stepsize and the previous
                # one
                if errt <= err[0]:
                    err[0] = errt
                    gradf[0] = a[jj, ii]

            if np.abs(a[ii, ii] - a[ii - 1, ii - 1] >= safe*err[0]):
                break

        # y direction
        a[0, 0] = (func(x[1], x[1] + hh[1]) -
                   func(x[0], x[1] - hh[1]))/(2.0*hh[1])
        err[1] = big
        for ii in range(ntab):
            # successive columns in the Neville tableau will go to smaller
            # stepsizes and higher orders of extrapolation
            hh[1] = hh[1]/con
            # trying a new smaller stepsize
            a[0][ii] = (func(x[1], x[1] + hh[1]) -
                        func(x[0], x[1] - hh[1]))/(2.0*hh[1])
            fac = con2

            for jj in range(1, ii + 1):
                a[jj, ii] = (a[jj - 1, ii]*fac - a[jj - 1, ii - 1])/(fac - 1.0)
                fac = con2*fac
                errt = np.max([np.abs(a[jj, ii] - a[jj - 1, ii]),
                               np.abs(a[jj, ii] - a[jj - 1, ii - 1])])
                # the error strategy is to compare each new extrapolation to
                # one order lower both at the present stepsize and the previous
                # one
                if errt <= err[1]:
                    err[1] = errt
                    gradf[1] = a[jj, ii]

            if np.abs(a[ii, ii] - a[ii - 1, ii - 1] >= safe*err[1]):
                break
        return gradf, err
    else:
        temp = x + h
        h = __doNothing(temp) - x
        gradf[0] = (func(x[0] + h[0], x[1]) - func(x[0] - h[0], x[1]))/(2*h[0])
        gradf[1] = (func(x[0], x[1] + h[1]) - func(x[0], x[1] - h[1]))/(2*h[1])
        return gradf
