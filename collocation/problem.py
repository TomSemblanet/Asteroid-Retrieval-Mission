#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  12 09:44:00 2020

@author: SEMBLANET Tom
"""

import numpy as np


class Problem:
    """ `Problem` class is the basic structure containing all the data used to define
        an optimal control problem. It's the user definition of the problem which will
        be passed to the transcription algorithm which will translate it into a parametric
        optimization, which can be solved by means of Non-Linear Programming (IPOPT or SNOPT).

        Parameters
        ----------
        n_states : int
            Number of states
        n_controls : int
            Number of controls
        n_path_con : int
            Number of path constraints
        n_event_con : int
            Number of event constraints
        n_f_par : int
            Number of free parameters
        n_nodes : int
            Number of nodes

        Attributes
        ----------
        prm : dict
            Problem parameters such as number of states, controls, path_constraints ...
        low_bnd : Boundaries
            Lower boundaries for the states, controls, free parameters and time
        upp_bnd : Boundaries
            Upper boundaries for the states, controls, free parameters and time
        initial_guess : Guess
            Initial guess object for the states, controls, free parameters and time
        end_point_cost : function
            Function returning the Mayer part of the cost function (optional if integrand_cost
            is given)
        integrand_cost : function
            Function returning the Legendre part of the cost function (optional if
            end_point_cost is given)
        dynamics : function
            Function returning the derivatives of the states at each time
        path_constraints : function
            Function returning the path constraints
        event_constraints : function
            Function returning the event constraints

    """

    def __init__(self, n_states, n_controls, n_path_con, n_event_con, n_f_par, n_nodes):
        """ Initialization of the Problem class """

        # Phase parameters dictionnary
        self.prm = {
            'n_states': n_states,
            'n_controls': n_controls,
            'n_f_par': n_f_par,
            'n_path_con': n_path_con,
            'n_event_con': n_event_con,
            'n_nodes': n_nodes
        }

        # Initialization of the states, controls, time and free parameters lower and
        # upper boundaries
        self.low_bnd = Boundaries(
            n_states, n_controls, n_f_par, n_path_con, n_event_con)
        self.upp_bnd = Boundaries(
            n_states, n_controls, n_f_par, n_path_con, n_event_con)

        # Initialization of the states, controls, time and free parameters initial guess
        self.initial_guess = Guess(
            n_states, n_controls, n_f_par, n_nodes)

    def setup(self, transcription_method):
        """ Completes the setup of the optimal control problem by calling
            the set_[...] functions and by setting parameters specific to the transcription method
            used

            Parameters
            ----------
            transcription_method : string
                Name of the transcription method used

        """

        # Setting the constants of the problem
        if hasattr(self, 'set_constants'):
            self.set_constants()

        # Setting the initial guess
        if hasattr(self, 'set_initial_guess'):
            self.set_initial_guess()

        # Setting the states, controls, free parameters, time, event and path constraints boundaries
        if hasattr(self, 'set_boundaries'):
            self.set_boundaries()
        if hasattr(self, 'set_events_constraints_boundaries'):
            self.set_events_constraints_boundaries()
        if hasattr(self, 'set_path_constraints_boundaries'):
            self.set_path_constraints_boundaries()

        # Setting of the number of collocation controls
        self.prm['n_controls_col'] = self.prm['n_controls'] if transcription_method == \
        'hermite-simpson' else 0

        # Setting of the lower, upper boundaries and initial guess for collocation controls
        self.low_bnd.controls_col = self.low_bnd.controls if self.prm['n_controls_col'] != 0 \
        else np.empty(shape=(0))
        self.upp_bnd.controls_col = self.upp_bnd.controls if self.prm['n_controls_col'] != 0 \
        else np.empty(shape=(0))
        self.initial_guess.controls_col = self.initial_guess.controls[:, :-1] if \
        self.prm['n_controls_col'] != 0  else np.empty(shape=(0, self.prm['n_nodes']-1))

        # Setting of the total number of variables
        self.prm['n_var'] = self.prm['n_nodes']*(self.prm['n_states']+self.prm['n_controls']) + \
            (self.prm['n_nodes']-1) * \
            self.prm['n_controls_col'] + self.prm['n_f_par'] + 2

        # Setting of the total number of constraints
        n = 1 if transcription_method in [
            'trapezoidal', 'hermite-simpson'] else 0
        self.prm['n_con'] = (self.prm['n_nodes']-n)*self.prm['n_states'] + \
        self.prm['n_event_con'] + self.prm['n_nodes']*self.prm['n_path_con']

        # Setting the transcription method name
        self.prm['tr_method_nm'] = transcription_method


class Boundaries:
    """ `Boundaries` class contains either the lower or upper boundaries of a problem
        decision variables vector (that contains states, controls, free parameters,
        final and initial time, path constraints, event constraints).
        Two `Boundaries` objects ar necessarily attached to a `Problem` object.

        Parameters
        ----------
        n_states : int
            Number of states
        n_controls : int
            Number of controls
        n_par : int
            Number of free parameters
        n_path_con : int
            Number of path constraints
        n_event_con : int
            Number of event constraints

        Attributes
        ----------
        states : array
            States lower or upper boundaries
        controls : array
            Controls lower or upper boundaries
        controls_col : array
            Collocation points controls lower or upper boundaries
        f_par : array
            Free parameters lower or upper boundaries
        path : array
            Path constraints lower or upper boundaries
        event : array
            Event constraints lower or upper boundaries
        ti : float
            Initial time lower or upper boundary
        tf : function
            Final time lower or upper boundary

        """

    def __init__(self, n_states, n_controls, n_par, n_path_con, n_event_con):
        """ Initialization of a `Boundaries` object """

        # States, Controls and Mid-Controls boundaries
        self.states = np.zeros(n_states)
        self.controls = np.zeros(n_controls)

        # Initial and Final time boundaries
        self.ti = 0.
        self.tf = 0.

        # Free parameters boundaries
        self.f_par = np.zeros(n_par)

        # Path and Event constraints boundaries
        self.path = np.zeros(n_path_con)
        self.event = np.zeros(n_event_con)


class Guess:
    """ `Guess` class contains the states, controls, free parameters and time-grid initial guess.
        A `Guess` object is necessarily attached to a `Problem` object.

        Parameters
        ----------
        n_states : int
            Number of states
        n_controls : int
            Number of controls
        n_par : int
            Number of free parameters
        n_nodes : int
            Number nodes

        Attributes
        ----------
        states : ndarray
            States initial guess
        controls : ndarray
            Controls initial guess
        controls_col : ndarray
            Collocation points controls initial guess
        f_prm : array
            Free parameters initial guess
        time : array
            Time grid initial guess

        """

    def __init__(self, n_states, n_controls, n_f_par, n_nodes):
        """ Initialization of a `Guess` object """

        self.states = np.ndarray(shape=(n_states, n_nodes))
        self.controls = np.ndarray(shape=(n_controls, n_nodes))
        self.f_prm = np.zeros(n_f_par)
        self.time = np.zeros(n_nodes)
