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
        n_st_path_con : int
            Number of states path constraints
        n_ct_path_con : int
            Number of controls path constraints
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

    def __init__(self, n_states, n_controls, n_st_path_con, n_ct_path_con, n_event_con, n_f_par, n_nodes):
        """ Initialization of the Problem class """

        # Phase parameters dictionnary
        self.prm = {
            'n_states': n_states,
            'n_controls': n_controls,
            'n_f_par': n_f_par,
            'n_st_path_con': n_st_path_con,
            'n_ct_path_con': n_ct_path_con,
            'n_event_con': n_event_con,
            'n_nodes': n_nodes
        }

        # Initialization of the states, controls, time and free parameters lower and
        # upper boundaries
        self.low_bnd = Boundaries(
            n_states, n_controls, n_f_par, n_st_path_con, n_ct_path_con, n_event_con)
        self.upp_bnd = Boundaries(
            n_states, n_controls, n_f_par, n_st_path_con, n_ct_path_con, n_event_con)

        # Initialization of the states, controls, time and free parameters initial guess
        self.initial_guess = Guess(
            n_states, n_controls, n_f_par, n_nodes)

    def setup(self):
        """ Completes the setup of the optimal control problem by calling
            the set_[...] functions and by setting additional and collocation point 
            initial guess

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


        # Setting of the initial guess for collocation controls 
        # (2 collocation node per segments)
        self.initial_guess.controls_col = np.repeat(self.initial_guess.controls[:, :-1], 2, 1)

        # Setting of the initial guess for additional states and controls 
        # (1 additional node per segments)
        self.initial_guess.states_add = self.initial_guess.states[:, :-1]
        self.initial_guess.controls_add = self.initial_guess.controls[:, :-1]

        # Setting of the total number of variables
        self.prm['n_var'] = self.prm['n_nodes']*(self.prm['n_states'] + self.prm['n_controls']) + \
            (self.prm['n_nodes'] - 1) * (3*self.prm['n_controls'] + self.prm['n_states']) + \
            self.prm['n_f_par'] + 2

        # Setting of the total number of constraints
        self.prm['n_con'] = 2*(self.prm['n_nodes']-1)*self.prm['n_states'] + \
        	(4*self.prm['n_nodes']-3)*self.prm['n_ct_path_con'] + \
            (2*self.prm['n_nodes']-1)*self.prm['n_st_path_con'] + \
            self.prm['n_event_con']

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
        n_st_path_con : int
            Number of states path constraints
        n_ct_path_con : int
            Number of controls path constraints
        n_event_con : int
            Number of event constraints

        Attributes
        ----------
        states : array
            States lower or upper boundaries
        controls : array
            Controls lower or upper boundaries
        f_par : array
            Free parameters lower or upper boundaries
        st_path : array
            States path constraints lower or upper boundaries
        ct_path : array
            Controls path constraints lower or upper boundaries
        event : array
            Event constraints lower or upper boundaries
        ti : float
            Initial time lower or upper boundary
        tf : function
            Final time lower or upper boundary

        """

    def __init__(self, n_states, n_controls, n_par, n_st_path_con, n_ct_path_con, n_event_con):
        """ Initialization of a `Boundaries` object """

        # States, Controls boundaries
        self.states = np.zeros(n_states)
        self.controls = np.zeros(n_controls)

        # Initial and Final time boundaries
        self.ti = 0.
        self.tf = 0.

        # Free parameters boundaries
        self.f_par = np.zeros(n_par)

        # Path and Event constraints boundaries
        self.st_path = np.zeros(n_st_path_con)
        self.ct_path = np.zeros(n_ct_path_con)

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
        states_add : ndarray
            Additional point states initial guess
        controls_add : ndarray
            Additional point controls initial guess
        f_prm : array
            Free parameters initial guess
        time : array
            Time grid initial guess

        """

    def __init__(self, n_states, n_controls, n_f_par, n_nodes):
        """ Initialization of a `Guess` object """

        # Initial guess of states and controls
        self.states = np.ndarray(shape=(n_states, n_nodes))
        self.controls = np.ndarray(shape=(n_controls, n_nodes))

        # Initial guess of collocation controls
        self.controls_col = np.ndarray(shape=(n_controls, 2*(n_nodes-1)))

        # Initial guess of additional states and controls
        self.states_add = np.ndarray(shape=(n_states, n_nodes-1))
        self.controls_add = np.ndarray(shape=(n_controls, n_nodes-1))

        # Initial guess for free parameters and time grid
        self.f_prm = np.zeros(n_f_par)
        self.time = np.zeros(n_nodes)
