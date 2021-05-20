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

from src.optimal_control.problem import Problem
from src.optimal_control.optimization import Optimization


class BrysonMaximumRange(Problem):
    """ Bryson Maximum Range : optimal control problem"""

    def __init__(self):
        """ Initialization of the `BrysonMaximumRange` class """
        n_states = 3
        n_controls = 2
        n_path_con = 1
        n_event_con = 4
        n_f_par = 0
        n_nodes = 100

        Problem.__init__(self, n_states, n_controls, n_path_con,
                         n_event_con, n_f_par, n_nodes)

    def set_constants(self):
        """ Setting of the problem constants """
        self.g = 1
        self.a = 0.5 * self.g

    def set_boundaries(self):
        """ Setting of the states, controls, free-parameters, initial and final times
                        boundaries """
        # States boundaries
        self.low_bnd.states[0] = -10
        self.upp_bnd.states[0] = 10

        self.low_bnd.states[1] = -10
        self.upp_bnd.states[1] = 10

        self.low_bnd.states[2] = -10
        self.upp_bnd.states[2] = 10

        # Controls boundaries
        self.low_bnd.controls[0] = -10
        self.upp_bnd.controls[0] = 10

        self.low_bnd.controls[1] = -10
        self.upp_bnd.controls[1] = 10

        # Initial and final times boundaries
        self.low_bnd.ti = self.upp_bnd.ti = 0
        self.low_bnd.tf = self.upp_bnd.tf = 2

    def event_constraints(self, xi, ui, xf, uf, ti, tf, f_prm):
        """ Computation of the events constraints """
        events = np.ndarray((self.prm['n_event_con'], 1),
                            dtype=cppad_py.a_double)

        x_i = xi[0]
        y_i = xi[1]
        v_i = xi[2]

        y_f = xf[1]

        events[0] = x_i
        events[1] = y_i
        events[2] = v_i
        events[3] = y_f

        return events

    def set_events_constraints_boundaries(self):
        """ Setting of the events constraints boundaries """
        self.low_bnd.event[0] = self.upp_bnd.event[0] = 0

        self.low_bnd.event[1] = self.upp_bnd.event[1] = 0

        self.low_bnd.event[2] = self.upp_bnd.event[2] = 0

        self.low_bnd.event[3] = self.upp_bnd.event[3] = 0.1

    def path_constraints(self, states, controls, f_par):
        """ Computation of the path constraints """
        paths = np.ndarray((self.prm['n_path_con'],
                            self.prm['n_nodes']), dtype=cppad_py.a_double)

        u1 = controls[0]
        u2 = controls[1]

        paths[0] = u1*u1 + u2*u2

        return paths

    def set_path_constraints_boundaries(self):
        """ Setting of the path constraints boundaries """
        self.low_bnd.path[0] = self.upp_bnd.path[0] = 1

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

        u1 = controls[0]
        u2 = controls[1]

        x_dot = v * u1
        y_dot = v * u2
        v_dot = self.a - self.g * u2

        dynamics[0] = x_dot
        dynamics[1] = y_dot
        dynamics[2] = v_dot

        return dynamics

    def end_point_cost(self, ti, xi, tf, xf, f_prm):
        """ Computation of the end point cost (Mayer term) """
        return - xf[0]

    def set_initial_guess(self):
        """ Setting of the initial guess for the states, controls, free-parameters
                        and time grid """
        # Time grid initial guess
        self.initial_guess.time = np.linspace(0, 2, self.prm['n_nodes'])

        # States initial guess
        self.initial_guess.states[0] = np.array(
            [0 for _ in self.initial_guess.time])
        self.initial_guess.states[1] = np.array(
            [0 for _ in self.initial_guess.time])
        self.initial_guess.states[2] = np.array(
            [0 for _ in self.initial_guess.time])

        # Controls initial guess
        self.initial_guess.controls[0] = np.array(
            [0 for _ in self.initial_guess.time])
        self.initial_guess.controls[1] = np.array(
            [0 for _ in self.initial_guess.time])


if __name__ == '__main__':

    # Instantiation of the problem
    problem = BrysonMaximumRange()

    # Instantiation of the optimization
    optimization = Optimization(problem=problem)

    # Launch of the optimization
    optimization.run()
