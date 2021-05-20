#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  14 09:50:23 2020

@author: SEMBLANET Tom

"""

import os
import cppad_py
import math
import numpy as np
import matplotlib.pyplot as plt

from src.problem import Problem
from src.optimization import Optimization


class Brachistochrone(Problem):
    """ Brachistochrone optimal control problem"""

    def __init__(self):
        """ Initialization of the `Brachistochrone` class """
        n_states = 3
        n_controls = 1
        n_st_path_con = 0
        n_ct_path_con = 0
        n_event_con = 5
        n_f_par = 0
        n_nodes = 300

        Problem.__init__(self, n_states, n_controls, n_st_path_con, n_ct_path_con, 
                         n_event_con, n_f_par, n_nodes)

    def set_constants(self):
        """ Setting of the problem constants """
        self.g = 9.8

    def set_boundaries(self):
        """ Setting of the states, controls, free-parameters, initial and final times
                        boundaries """
        # States boundaries
        self.low_bnd.states[0] = 0
        self.upp_bnd.states[0] = 20

        self.low_bnd.states[1] = 0
        self.upp_bnd.states[1] = 20

        self.low_bnd.states[2] = 0
        self.upp_bnd.states[2] = 20

        # Controls boundaries
        self.low_bnd.controls[0] = 0
        self.upp_bnd.controls[0] = 2 * np.pi

        # Initial and final times boundaries
        self.low_bnd.ti = self.upp_bnd.ti = 0
        self.low_bnd.tf = 0
        self.upp_bnd.tf = 10

    def event_constraints(self, xi, ui, xf, uf, ti, tf, f_prm):
        """ Computation of the events constraints """
        events = np.ndarray((self.prm['n_event_con'], 1),
                            dtype=cppad_py.a_double)

        # Initial values of states are constrained
        x_i = xi[0]
        y_i = xi[1]
        v_i = xi[2]

        # Final values of states n°1 and n°2 are constrained
        x_f = xf[0]
        y_f = xf[1]

        events[0] = x_i
        events[1] = y_i
        events[2] = v_i
        events[3] = x_f
        events[4] = y_f

        return events

    def set_events_constraints_boundaries(self):
        """ Setting of the events constraints boundaries """
        self.low_bnd.event[0] = self.upp_bnd.event[0] = 0

        self.low_bnd.event[1] = self.upp_bnd.event[1] = 0

        self.low_bnd.event[2] = self.upp_bnd.event[2] = 0

        self.low_bnd.event[3] = self.upp_bnd.event[3] = 2

        self.low_bnd.event[4] = self.upp_bnd.event[4] = 2

    def dynamics(self, states, controls, f_prm, expl_int=False):
        """ Computation of the states derivatives """
        if expl_int == False:
            dynamics = np.ndarray(
                (states.shape[0], states.shape[1]), dtype=cppad_py.a_double)
        else:
            dynamics = np.zeros(len(states))

        x = states[0]
        y = states[1]
        v = states[2]

        theta = controls[0]

        x_dot = v * np.sin(theta)
        y_dot = v * np.cos(theta)
        v_dot = self.g * np.cos(theta)

        dynamics[0] = x_dot
        dynamics[1] = y_dot
        dynamics[2] = v_dot

        return dynamics

    def end_point_cost(self, ti, xi, tf, xf, f_prm):
        """ Computation of the end point cost (Mayer term) """
        return tf

    def set_initial_guess(self):
        """ Setting of the initial guess for the states, controls, free-parameters
                        and time grid """

        # Time
        self.initial_guess.time = np.linspace(0, 4, self.prm['n_nodes'])

        # States
        self.initial_guess.states = np.ndarray(
            shape=(self.prm['n_states'], self.prm['n_nodes']))

        self.initial_guess.states[0] = np.linspace(0, 1, self.prm['n_nodes'])
        self.initial_guess.states[1] = np.linspace(0, 1, self.prm['n_nodes'])
        self.initial_guess.states[2] = np.linspace(0, 1, self.prm['n_nodes'])

        # Controls
        self.initial_guess.controls = np.ndarray(
            shape=(self.prm['n_controls'], self.prm['n_nodes']))

        self.initial_guess.controls[0] = np.array(
            [1 for _ in range(len(self.initial_guess.time))])

        # Time grid initial guess
        self.initial_guess.time = np.linspace(0, 4, self.prm['n_nodes'])


if __name__ == '__main__':

    # Instantiation of the problem
    problem = Brachistochrone()

    # Instantiation of the optimization
    optimization = Optimization(problem=problem)

    # Launch of the optimization
    optimization.run()
