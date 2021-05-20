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
    if (abs(scl_x[-1] - out_range[1]) > 1e-12 or abs(scl_x[0] - out_range[0]) > 1e-12):
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


def make_decision_variable_vector(states, controls, states_add, controls_add, controls_col, ti, tf, prm, pb_prm):
    """ Packs the states, control, free parameters, initial and final time informations 
            into a decision variable vector 

        Parameters
        ----------
        states : ndarray
            States matrix
        controls : ndarray
            Controls matrix
        states_add : ndarray
            Additional node states matrix
        controls_add : ndarray
            Additional node controls matrix
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

    # Conversion of states and controls matrices into 1D arrays
    states_vec = numpy.concatenate([states[:, i]
                                    for i in range(pb_prm['n_nodes'])])
    controls_vec = numpy.concatenate(
        [controls[:, i] for i in range(pb_prm['n_nodes'])])

    # Conversion of additional nodes states and controls matrice 
    # into 1D arrays
    states_add_vec = numpy.concatenate([states_add[:, i] for i in range(pb_prm['n_nodes']-1)])
    controls_add_vec = numpy.concatenate([controls_add[:, i] for i in range(pb_prm['n_nodes']-1)])

    # Conversion of collocation nodes controls matrix into 1D array
    controls_col_vec = numpy.concatenate([controls_col[:, i] for i in range(2*(pb_prm['n_nodes']-1))])

    return numpy.concatenate((states_vec, controls_vec, states_add_vec, controls_add_vec, controls_col_vec, prm, time_vec))


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
        states_add : ndarray
            States at additional points matrix
        controls_add : ndarray
        	Controls at additional points matrix
        controls_col : ndarray
        	Controls at collocation points matrix

        """

    # Extraction and construction of state matrix
    states_vec = decision_variable_vector[:(prm['n_nodes'] * prm['n_states'])]
    states = numpy.transpose(numpy.reshape(
        states_vec, (prm['n_nodes'], prm['n_states'])))

    # Extraction and construction of control matrix
    controls_vec = decision_variable_vector[
        prm['n_nodes'] * prm['n_states']:prm['n_nodes'] * (prm['n_states']+prm['n_controls'])]
    controls = numpy.transpose(numpy.reshape(
        controls_vec, (prm['n_nodes'], prm['n_controls'])))

    # Extraction and construction of additional states matrix
    states_add_vec = decision_variable_vector[
    	prm['n_nodes'] * (prm['n_states'] + prm['n_controls']):prm['n_nodes'] * (prm['n_states'] + \
    	prm['n_controls']) + (prm['n_nodes']-1)*prm['n_states']]
    states_add = numpy.transpose(numpy.reshape(
        states_add_vec, (prm['n_nodes']-1, prm['n_states'])))

    # Extraction and construction of additional controls matrix
    controls_add_vec = decision_variable_vector[
    	prm['n_nodes'] * (prm['n_states'] + prm['n_controls']) + (prm['n_nodes']-1)*prm['n_states']:\
    	prm['n_nodes'] * (prm['n_states'] + prm['n_controls']) + (prm['n_nodes']-1)*(prm['n_states']+prm['n_controls'])]
    controls_add = numpy.transpose(numpy.reshape(
        controls_add_vec, (prm['n_nodes']-1, prm['n_controls'])))

    # Extraction and construction of collocations controls matrix
    controls_col_vec = decision_variable_vector[
    	prm['n_nodes'] * (prm['n_states'] + prm['n_controls']) + (prm['n_nodes']-1)*(prm['n_states']+prm['n_controls']):\
    	prm['n_nodes'] * (prm['n_states'] + prm['n_controls']) + (prm['n_nodes']-1)*(prm['n_states']+prm['n_controls']) + \
    	2*(prm['n_nodes']-1)*prm['n_controls']]
    controls_col = numpy.transpose(numpy.reshape(
        controls_col_vec, (2*(prm['n_nodes']-1), prm['n_controls'])))

    # Extraction of the free parameters
    f_prm = decision_variable_vector[-(2+prm['n_f_par']):-2]

    # Extraction of initial and final times
    ti = decision_variable_vector[-2]
    tf = decision_variable_vector[-1]

    return ti, tf, f_prm, states, controls, states_add, controls_add, controls_col

def build_final_controls_vec(generic_ctrl, additional_ctrl, collocation_ctrl):
    """ Concatenates the generic, additional and collocation controls into one 
        control array

        Parameters
        ----------
        generic_ctrl : ndarray
            Controls at generic nodes
        additional_ctrl : ndarray
            Controls at additional nodes
        collocation_ctrl : ndarray
            Controls at collocation nodes

        Returns
        -------
        ctrl : ndarray
            Control array

    """
    ctrl = numpy.ndarray(shape=(generic_ctrl.shape[0], generic_ctrl.shape[1] + additional_ctrl.shape[1] + \
                            collocation_ctrl.shape[1]))

    for k in range(generic_ctrl.shape[1]):
        ctrl[:, 4*k] = generic_ctrl[:, k]
        if k != len(generic_ctrl[0])-1:
            ctrl[:, 4*k+1] = collocation_ctrl[:, 2*k]
            ctrl[:, 4*k+2] = additional_ctrl[:, k]
            ctrl[:, 4*k+3] = collocation_ctrl[:, 2*k+1]

    return ctrl


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


def explicit_integration(time, states, controls, f_par, dynamics, interp_func):
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

    def dynamic_redef(t, y, u, f_par):
        """ Redefinition of the dynamics equation """

        # Control interpolation at time `t`
        u_t = interp_func(u, t)

        return dynamics(y, u_t, f_par, expl_int=True)

    # Extraction of the initial conditions
    y0 = states

    # Explicit integration using scipy.solve_ivp module
    explicit_int = solve_ivp(fun=dynamic_redef, t_span=(time[0], time[-1]), y0=y0, t_eval=time,
                             args=(controls, f_par), method='DOP853', rtol=3e-14, atol=1e-14)
    intgr_states = explicit_int.y

    return intgr_states
