#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  12 11:05:12 2020

@author: SEMBLANET Tom

"""

import numpy as np
import cppad_py
import math as mt

from scipy import interpolate

from collocation.GL_V.src import utils


class GL5:
    """ `GL5` class implements the Gauss-Lobatto transcription method of order 5 allowing
        to compute the defects constraints values and to approximate the value of an integrand using
        Gauss-Lobatto (V) quadrature.

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

        # Coefficients used to compute the collocation states 
        self.states_col_coeff = np.array([[39*mt.sqrt(21)+231, 224, -39*mt.sqrt(21)+231, 3*mt.sqrt(21)+21, -16*mt.sqrt(21), 3*mt.sqrt(21)-21],
        								   [-39*mt.sqrt(21)+231, 224, 39*mt.sqrt(21)+231, -3*mt.sqrt(21)+21, 16*mt.sqrt(21), -3*mt.sqrt(21)-21]])

        # Coefficients used to compute the defects at collocation points
        self.defects_col_coeff = np.array([[32*mt.sqrt(21)+180, -64*mt.sqrt(21), 32*mt.sqrt(21)-180, 9+mt.sqrt(21), 98, 64, 9-mt.sqrt(21)],
        								   [-32*mt.sqrt(21)+180, 64*mt.sqrt(21), -32*mt.sqrt(21)-180, 9-mt.sqrt(21), 98, 64, 9+mt.sqrt(21)]])

        self.defects = np.ndarray(
            (options['n_states'], options['n_nodes']-1), dtype=cppad_py.a_double)

    def compute_defects(self, states, controls, states_add, controls_add, controls_col, f_prm, f, sc_fac):
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

        # Computation of the derivatives matrix at generic points
        F = sc_fac * f(states, controls, f_prm)

        # Computation of the derivatives matrix at the additional points
        F_add = sc_fac * f(states_add, controls_add, f_prm)

        # Computation of the states at collocation nodes
        states_col_1, states_col_2 = self.compute_states_col(states, controls, states_add, controls_add, F, F_add)

        # Computation of derivatives at collocation points
        F_col_1 = sc_fac * f(states_col_1, controls_col[:, 0::2], f_prm)
        F_col_2 = sc_fac * f(states_col_2, controls_col[:, 1::2], f_prm)

        # Computation of defects matrix
        defects_1 = 1/360 * (self.defects_col_coeff[0, 0]*states[:, :-1] + self.defects_col_coeff[0, 1]*states_add[:, :] + \
        						self.defects_col_coeff[0, 2]*states[:, 1:] + \
        						self.options['h'] * (self.defects_col_coeff[0, 3]*F[:, :-1] + self.defects_col_coeff[0, 4]*F_col_1[:, :] + \
        						self.defects_col_coeff[0, 5]*F_add[:, :] + self.defects_col_coeff[0, 6]*F[:, 1:]))
       	defects_2 = 1/360 * (self.defects_col_coeff[1, 0]*states[:, :-1] + self.defects_col_coeff[1, 1]*states_add[:, :] + \
        						self.defects_col_coeff[1, 2]*states[:, 1:] + \
        						self.options['h'] * (self.defects_col_coeff[1, 3]*F[:, :-1] + self.defects_col_coeff[1, 4]*F_col_2[:, :] + \
        						self.defects_col_coeff[1, 5]*F_add[:, :] + self.defects_col_coeff[1, 6]*F[:, 1:]))

       	self.defects = np.ravel([defects_1,defects_2],order="F").reshape(len(defects_1),2*len(defects_1[0]))

        return self.defects


    def compute_states_col(self, states, controls, states_add, controls_add, F, F_add):
        """ Computation of the states at the collocation points

            Parameters
            ----------
            states : ndarray
                Matrix of the states at the generic points
            controls : ndarray
                Matrix of the controls at the generic points
            states_add : ndarray
            	Matrix of the states at the additional points
            controls_add : ndarray
            	Matrix of the controls at the additional points
            F : ndarray
            	Matrix of the derivatives at the generic points
            F_add : ndarray
            	Matrix of the derivatives at the additional points

            Returns
            -------
            states_col : ndarray
            	Matrix of the states at the additional points

        """

        # Computation of the states at collocation points
        states_col_1 = 1/686 * (self.states_col_coeff[0, 0]*states[:, :-1] + self.states_col_coeff[0, 1]*states_add[:, :] + \
        						self.states_col_coeff[0, 2]*states[:, 1:] + \
        						self.options['h'] * (self.states_col_coeff[0, 3]*F[:, :-1] + self.states_col_coeff[0, 4]*F_add[:, :] + \
        						self.states_col_coeff[0, 5]*F[:, 1:]))

        states_col_2 = 1/686 * (self.states_col_coeff[1, 0]*states[:, :-1] + self.states_col_coeff[1, 1]*states_add[:, :] + \
        						self.states_col_coeff[1, 2]*states[:, 1:] + \
        						self.options['h'] * (self.states_col_coeff[1, 3]*F[:, :-1] + self.states_col_coeff[1, 4]*F_add[:, :] + \
        						self.states_col_coeff[1, 5]*F[:, 1:]))
        
        # Problem occurs if the states_col elements type is <class 'float'>
        if str(type(states_col_1[0, 0])) == "<class 'float'>":
            states_col_1 = states_col_1.astype('float64')
            states_col_2 = states_col_2.astype('float64')

        return states_col_1, states_col_2


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

    def build_interpolation_func(self, time, controls):
        """ Construction of the controls interpolation function using Scipy's method interp1d

            Parameters
            ----------
            time : array
                Time grid array
            controls : ndarray
                Matrix of the controls

        """

        # List containing the interpolation functions for each controls
        f_cx = list()

        # Construction of the interpolation functions
        for ct_k in controls:

            # Interpolation function of the k-ieme control
            f_ck = interpolate.interp1d(time, ct_k, kind='cubic')
            f_cx.append(f_ck)

        self.f_cx = f_cx

    def interpolate_ctrl(self, controls, t):
        """ Controls interpolation function, used for explicit integration

            Parameters
            ----------
            controls : ndarray
                Matrix of the controls
            t : float
                Interpolation time

            Returns
            -------
            array
                Array of interpolated controls

        """
        return np.array([self.f_cx[i](t) for i in range(len(controls))])
