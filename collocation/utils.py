#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  12 16:50:53 2020

@author: SEMBLANET Tom

"""

import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import numpy
import math


def scale_time(x, out_range=(-1, 1)):
    """ Scales an array to [min(out_range), max(out_range)] 

        Parameters
        ----------
        x : array
            Array to scale
        out_range : tuple
            Lower and upper values of the scaled array

        Returns
        -------
        scl_x : array
            Scaled array

    """

    domain = min(x), max(x)

    # Scaling of the `x` input
    y = (x - (domain[1] + domain[0]) / 2.) / (domain[1] - domain[0])
    scl_x = y * (out_range[1] - out_range[0]) + \
        (out_range[1] + out_range[0]) / 2.

    # Round error can occur, making that x isn't exactly in the
    # `out_range` range. A rescaling is then called.
    if ((scl_x[-1] == out_range[1]) == False or (scl_x[0] == out_range[0]) == False):
        scl_x = scale_time(scl_x, out_range)

    return scl_x


def retrieve_time_grid(h_vec, ti, tf):
    """ Retrieves the initial time grid. Used after an optimization process
        during which the time grid has been scaled to [-1, 1] 

        Parameters
        ----------
        h_vec : array
            Array of the scaled time-steps
        ti : float
            Unscaled initial time
        tf : float
            Unscaled final time

        Returns
        -------
        scaled_time_grid : array
            Unscaled time grid

    """

    tm_grid = numpy.array([-1.])
    x = -1.
    for h_ in h_vec:
        x += h_
        tm_grid = numpy.append(tm_grid, x)

    scaled_time_grid = scale_time(tm_grid, (ti, tf))
    return scaled_time_grid


def make_decision_variable_vector(states, controls, controls_col, ti, tf, prm, pb_prm):
    """ Packs the states, control, free parameters, initial and final time informations 
            into a decision variable vector 

        Parameters
        ----------
        states : ndarray
            States matrix
        controls : ndarray
            Controls matrix
        controls_col : ndarray
            Controls at collocation points matrix
        ti : float
            Initial time
        tf : float
            Final time
        prm : array
            Free parameters array
        pb_prm : dict
            Problem parameters

        Returns
        -------
        array
            Decision variables vector

        """
    time_vec = numpy.array([ti, tf])
    states_vec = numpy.concatenate([states[:, i]
                                    for i in range(pb_prm['n_nodes'])])
    controls_vec = numpy.concatenate(
        [controls[:, i] for i in range(pb_prm['n_nodes'])])

    if pb_prm['tr_method_nm'] == 'hermite-simpson':
        controls_col_vec = numpy.concatenate(
            [controls_col[:, i] for i in range(pb_prm['n_nodes']-1)])
    else:
        controls_col_vec = numpy.empty(0)

    return numpy.concatenate((states_vec, controls_vec, controls_col_vec, prm, time_vec))


def unpack_decision_variable_vector(decision_variable_vector, prm):
    """ Unpacks a Decision Variable Vector into states and controls
        matrices, free parameters array and initial/final times informations 

        Parameters
        ----------
        decision_variable_vector : array
            Decision variables vector
        prm : dict
            Problem parameters

        Returns
        -------
        ti : float
            Initial time
        tf : float
            Final time
        f_prm : array
            Free parameters array
        states : ndarray
            States matrix
        controls : ndarray
            Controls matrix
        controls_col : ndarray
            Controls at collocation points matrix

        """

    # Extraction of initial and final times
    ti = decision_variable_vector[-2]
    tf = decision_variable_vector[-1]

    # Extraction of the free parameters
    f_prm = decision_variable_vector[-(2+prm['n_f_par']):-2]

    # Extraction and construction of state matrix
    states_vec = decision_variable_vector[:(prm['n_nodes'] * prm['n_states'])]
    states = numpy.transpose(numpy.reshape(
        states_vec, (prm['n_nodes'], prm['n_states'])))

    # Extraction and construction of control matrix
    controls_vec = decision_variable_vector[
        prm['n_nodes'] * prm['n_states']:-((prm['n_nodes']-1)*prm['n_controls_col'] + 2 + prm['n_f_par'])]
    controls = numpy.transpose(numpy.reshape(
        controls_vec, (prm['n_nodes'], prm['n_controls'])))

    # Exctraction and construction of col-points control matrix
    if prm['tr_method_nm'] == 'hermite-simpson':
        controls_col_vec = decision_variable_vector[
            prm['n_nodes'] * (prm['n_states'] + prm['n_controls']):-2-prm['n_f_par']]
        controls_col = numpy.transpose(numpy.reshape(
            controls_col_vec, (prm['n_nodes']-1, prm['n_controls_col'])))
    else:
        controls_col = numpy.empty(0)

    return ti, tf, f_prm, states, controls, controls_col


def classic_display(time, states, controls, states_g=None, controls_g=None):
    """ Displays the optimization results 

        Parameters
        ----------
        time : array
            Time-grid array
        states : ndarray
            States matrix
        controls : ndarray
            Controls matrix
        states_g : array
            States labels
        controls_g : array
            Controls labels

    """

    # Plots the states (and optionaly the initial guess)
    plt.subplot(211)
    for k, st_ in enumerate(states):
        plt.plot(time, st_, '-', label=('x'+str(k)))

    if states_g is not None:
        for k, st_ in enumerate(states_g):
            plt.plot(time, st_, '-', label=('x'+str(k)+' (g)'))

    plt.legend()
    plt.grid()

    plt.subplot(212)
    for k, ct_ in enumerate(controls):
        plt.plot(time, ct_, '-', label=('u'+str(k)))

    if controls_g is not None:
        for k, ct_ in enumerate(controls_g):
            plt.plot(time, ct_, '-', label=('u'+str(k)+' (g)'))

    plt.legend()
    plt.grid()

    plt.show()


def explicit_integration(time, states, controls, controls_col, f_par, dynamics, interp_func):
    """ Run an explicit integration of the ODEs with the controls 
        returned by the optimizer 

        Parameters
        ----------
        time : array
            Time-grid array
        states : ndarray
            States matrix
        controls : ndarray
            Controls matrix
        controls_col : ndarray
            Controls at collocation points matrix
        f_par : array
            Free parameters array
        dynamics : function
            Dynamics function (user defined)
        interp_func : function  
            Controls interpolation function specific to the transcription method used
            (see Trapezoidal, HermiteSimpson or Pseudospectrals classes)

        Returns
        -------
        intgr_states : ndarray
            Integrated states matrix

        """

    def dynamic_redef(t, y, u, u_col, f_par, time):
        """ Redefinition of the dynamics equation """

        # Control interpolation at time `t`
        u_t = interp_func(time, u, u_col, t)

        return dynamics(y, u_t, f_par, expl_int=True)

    # Extraction of the initial conditions
    y0 = states

    # Explicit integration using scipy.solve_ivp module
    explicit_int = solve_ivp(fun=dynamic_redef, t_span=(time[0], time[-1]), y0=y0, t_eval=time,
                             args=(controls, f_par, time), method='DOP853', rtol=3e-14, atol=1e-14)
    intgr_states = explicit_int.y

    return intgr_states
