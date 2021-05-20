#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  12 11:05:12 2020

@author: SEMBLANET Tom

"""

from scipy import interpolate
import numpy as np
import cppad_py

from collocation import utils

# pylint: disable=invalid-name


class Trapezoidal:
    """ `Trapezoidal` class implements the Gauss-Lobatto transcription method of order 2 allowing
        to compute the defects constraints values and to approximate the value of an integrand using
        trapezoidal quadrature.

        Parameters
        ----------
        options : dict
            Transcription and Optimization options dictionnary

        Attributs
        ---------
        options : dict
            Transcription and Optimization options dictionnary
        defects : ndarray
            Matrix of the defects constraints values

    """

    def __init__(self, options):
        """ Initialization of the `Trapezoidal` class """
        self.options = options

        self.defects = np.ndarray(
            (options['n_states'], options['n_nodes']-1), dtype=cppad_py.a_double)

    def compute_defects(self, states, controls, f_prm, f, sc_fac):
        """ Computation of the defects constraints values using trapezoidal method.

            Parameters
            ----------
            states : ndarray
                Matrix of states
            controls : ndarray
                Matrix of the controls
            f_prm : array
                Array of the free parameters
            f : function
                Function of the dynamics
            sc_fac : float
                Scale factor

            Return
            ------
            defects : ndarray
                Matrix of the defects constraints values

        """

        F = sc_fac * f(states, controls, f_prm)

        # Computation of the defects matrix
        self.defects = states[:, 1:] - states[:, :-1] - \
            self.options['h']/2 * (F[:, 1:] + F[:, :-1])

        return self.defects

    def quadrature(self, func_values):
        """ Approximates the integrand of a funtion using trapezoidal quadrature.

            Parameters
            ----------
            func_values : array
                Values of the function at the nodes of the time grid

            Return
            ------
            sum : float
                Integrand approximation value

        """

        sum_ = 0
        for k in range(len(func_values)-1):
            sum_ += self.options['h'][k]/2 * \
                (func_values[k]+func_values[k+1])

        return sum_

    def nodes_adaptation(self, x_i, u_i, t_i):
        """ Scales the time grid so it belongs to the interval [-1, 1]
            and computes the time-step array.

            Parameters
            ----------
            x_i : ndarray
                Matrix of the initial guess states
            u_i : ndarray
                Matrix of the initial guess controls
            t_i : array
                Array of the initial guess time grid

            Returns
            -------
            x : ndarray
                Matrix of the states
            u : ndarray
                Matrix of the controls
            h : array
                Array of the time-steps

        """

        # Time scaling
        scl_time = utils.scale_time(t_i)

        # Computation of time-steps
        h = scl_time[1:] - scl_time[:-1]

        # Save the time-steps array in the object's options
        # dictionnary
        self.options['h'] = h

        # States and controls values remain the same, the time
        # is just scaled and we don't interpolate at different points
        x = x_i
        u = u_i

        return x, u, h

    def build_interpolation_func(self, time, controls, controls_col):
        """ Construction of the controls interpolation function using Scipy's method interp1d

            Parameters
            ----------
            time : array
                Time grid array
            controls : ndarray
                Matrix of the controls
            controls_col : ndarray
                Matrix of the controls at collocation points

        """

        # List containing the interpolation functions for each controls
        f_cx = list()

        # Construction of the interpolation functions
        for ct in controls_col:
            # Interpolation function of the k-ieme control
            f_ck = interpolate.interp1d(time, ct)
            f_cx.append(f_ck)

        self.f_cx = f_cx

    def interpolate_ctrl(self, time, controls, t):
        """ Controls interpolation function, used for explicit integration

            Parameters
            ----------
            time : array
                Time grid array
            controls : ndarray
                Matrix of the controls
            controls_col : ndarray
                Matrix of the controls at collocation points
            t : float
                Interpolation time

            Returns
            -------
            array
                Array of interpolated controls

        """
        return np.array([self.f_cx[i](t) for i in range(len(controls))])


class HermiteSimpson:
    """ `HermiteSimpson` class implements the Gauss-Lobatto transcription method of order 4 allowing
        to compute the defects constraints values and to approximate the value of an integrand using
        trapezoidal quadrature.

        Parameters
        ----------
        options : dict
            Transcription and Optimization options dictionnary

        Attributs
        ---------
        options : dict
            Transcription and Optimization options dictionnary
        defects : ndarray
            Matrix of the defects constraints values

    """

    def __init__(self, options):
        """ Initialization of the `HermiteSimpson` class """
        self.options = options

        self.defects = np.ndarray(
            (self.options['n_states'], self.options['n_nodes']-1), dtype=cppad_py.a_double)

    def compute_states_col(self, states, controls, f_prm, f, sc_fac):
        """ Computation of the states at the collocation points

            Parameters
            ----------
            states : ndarray
                Matrix of the states at the generic points
            controls : ndarray
                Matrix of the controls at the generic points
            f_prm : array
                Array of the free parameters
            f : function
                Function of the dynamic
            sc_fac : float
                Time scaling factor

            Returns
            -------
            states_col : ndarray
                Matrix of the states at the collocation points

        """

        # Computation of the derivatives matrix
        F = sc_fac * f(states, controls, f_prm)

        # Computation of the states at collocation points
        states_col = .5 * (states[:, 1:] + states[:, :-1]) + \
            self.options['h']/8 * (F[:, :-1] - F[:, 1:])

        # if(str(type(states_col[0, 0])) == 'float')
        if str(type(states_col[0, 0])) == "<class 'float'>":
            states_col = states_col.astype('float64')

        return states_col

    def compute_defects(self, states, controls, controls_col, f_prm, f, sc_fac):
        """ Computation of the defects constraints values using hermite-simpson method.

            Parameters
            ----------
            states : ndarray
                Matrix of states
            controls : ndarray
                Matrix of the controls
            controls_col : ndarray
                Matrix of the controls at collocation nodes
            f_prm : array
                Array of the free parameters
            f : function
                Function of the dynamics
            sc_fac : float
                Scale factor

            Return
            ------
            defects : ndarray
                Matrix of the defects constraints values

        """

        # Computation of derivatives at grid-points
        F = sc_fac * f(states, controls, f_prm)

        # Computation of states at col-points
        states_col = .5 * (states[:, 1:] + states[:, :-1]) + \
            self.options['h']/8 * (F[:, :-1] - F[:, 1:])

        # Problem occurs if the states_col elements type is <class 'float'>
        if str(type(states_col[0, 0])) == "<class 'float'>":
            states_col = states_col.astype('float64')

        # Computation of derivatives at col-points
        F_col = sc_fac * f(states_col, controls_col, f_prm)

        # Computation of defects matrix
        self.defects = states[:, 1:] - states[:, :-1] - \
            self.options['h']/6 * (F[:, :-1] + 4*F_col + F[:, 1:])

        return self.defects

    def quadrature(self, f_val, f_val_col):
        """ Approximates the integrand of a funtion using hermite-simpson quadrature.

            Parameters
            ----------
            f_val : array
                Values of the function at the nodes of the time grid
            f_val_val : array
                Values of the function at the collocation nodes

            Return
            ------
            sum : float
                Integrand approximation value

        """

        sum_ = 0
        for k in range(len(f_val)-1):
            sum_ += self.options['h'][k]/6 * \
                (f_val[k] + 4*f_val_col[k] + f_val[k+1])

        return sum_

    def nodes_adaptation(self, x_i, u_i, t_i):
        """ Scales the time grid so it belongs to the interval [-1, 1]
            and computes the time-step array.

            Parameters
            ----------
            x_i : ndarray
                Matrix of the initial guess states
            u_i : ndarray
                Matrix of the initial guess controls
            t_i : array
                Array of the initial guess time grid

            Returns
            -------
            x : ndarray
                Matrix of the states
            u : ndarray
                Matrix of the controls
            h : array
                Array of the time-steps

        """

        # Time scaling
        self.scl_time = utils.scale_time(t_i)

        # Computation of time-steps
        h = self.scl_time[1:] - self.scl_time[:-1]

        # Save the time-steps array in the object's options dictionnary
        self.options['h'] = h

        # States and controls values remain the same
        x = x_i
        u = u_i

        return x, u, h

    def build_interpolation_func(self, time, controls, controls_col):
        """ Construction of the controls interpolation function using Scipy's method interp1d

            Parameters
            ----------
            time : array
                Time grid array
            controls : ndarray
                Matrix of the controls
            controls_col : ndarray
                Matrix of the controls at collocation points

        """

        # List containing the interpolation functions for each controls
        f_cx = list()

        # Creation of the collocation points
        h = time[1:] - time[:-1]
        time_col = [t_ + h_/2 for t_, h_ in zip(time[:-1], h)]

        time_ = np.dstack((time[:-1], time_col)).flatten()
        time_ = np.concatenate((time_, [time[-1]]))

        # Construction of the interpolation functions
        for (ct, ct_m) in zip(controls, controls_col):
            ck_ = np.dstack((ct[:-1], ct_m)).flatten()
            ck_ = np.concatenate((ck_, [ct[-1]]))

            f_ck = interpolate.interp1d(time_, ck_)

            # Interpolation function of the k-ieme control
            f_ck = interpolate.interp1d(time_, ck_, kind='cubic')
            f_cx.append(f_ck)

        self.f_cx = f_cx

    def interpolate_ctrl(self, time, controls, t):
        """ Controls interpolation function, used for explicit integration

            Parameters
            ----------
            time : array
                Time grid array
            controls : ndarray
                Matrix of the controls
            controls_col : ndarray
                Matrix of the controls at collocation points
            t : float
                Interpolation time

            Returns
            -------
            array
                Array of interpolated controls

        """
        return np.array([self.f_cx[i](t) for i in range(len(controls))])
