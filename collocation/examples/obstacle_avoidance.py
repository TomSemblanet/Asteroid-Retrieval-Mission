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


class ObstacleAvoidance(Problem):
    """ Obstacle Avoidance : optimal control problem """

    def __init__(self):
        """ Initialization of the `ObstacleAvoidance` class """
        n_states = 2
        n_controls = 1
        n_path_con = 2
        n_event_con = 4
        n_f_par = 0
        n_nodes = 300

        Problem.__init__(self, n_states, n_controls, n_path_con,
                         n_event_con, n_f_par, n_nodes)

    def set_constants(self):
        """ Setting of the problem constants """
        self.V = 2.138

    def set_boundaries(self):
        """ Setting of the states, controls, free-parameters, initial and final times
                        boundaries """
        # States boundaries
        self.low_bnd.states[0] = - 20
        self.upp_bnd.states[0] = 20

        self.low_bnd.states[1] = - 20
        self.upp_bnd.states[1] = 20

        # Controls boundaries
        self.low_bnd.controls[0] = - 5
        self.upp_bnd.controls[0] = 5

        # Initial and final times boundaries
        self.low_bnd.ti = self.upp_bnd.ti = 0
        self.low_bnd.tf = self.upp_bnd.tf = 1

    def event_constraints(self, xi, ui, xf, uf, ti, tf, f_prm):
        """ Computation of the events constraints """
        events = np.ndarray((self.prm['n_event_con'], 1),
                            dtype=cppad_py.a_double)

        x_initial = xi[0]
        y_initial = xi[1]

        x_final = xf[0]
        y_final = xf[1]

        events[0] = x_initial
        events[1] = y_initial
        events[2] = x_final
        events[3] = y_final

        return events

    def set_events_constraints_boundaries(self):
        """ Setting of the events constraints boundaries """
        self.low_bnd.event[0] = self.upp_bnd.event[0] = 0

        self.low_bnd.event[1] = self.upp_bnd.event[1] = 0

        self.low_bnd.event[2] = self.upp_bnd.event[2] = 1.2

        self.low_bnd.event[3] = self.upp_bnd.event[3] = 1.6

    def path_constraints(self, states, controls, f_par):
        """ Computation of the path constraints """
        paths = np.ndarray((self.prm['n_path_con'],
                            self.prm['n_nodes']), dtype=cppad_py.a_double)

        x = states[0]
        y = states[1]

        paths[0] = (x-0.4)*(x-0.4) + (y-0.5)*(y-0.5)
        paths[1] = (x-0.8)*(x-0.8) + (y-1.5)*(y-1.5)

        return paths

    def set_path_constraints_boundaries(self):
        """ Setting of the path constraints boundaries """
        self.low_bnd.path[0] = 0.1
        self.upp_bnd.path[0] = 10

        self.low_bnd.path[1] = 0.1
        self.upp_bnd.path[1] = 10

    def dynamics(self, states, controls, f_prm, expl_int=False):
        """ Computation of the states derivatives """
        if expl_int == False:
            dynamics = np.ndarray(
                (states.shape[0], states.shape[1]), dtype=cppad_py.a_double)
        else:
            dynamics = np.zeros(len(states))

        # Theta angle
        theta = controls[0]

        x_dot = self.V * np.cos(theta)
        y_dot = self.V * np.sin(theta)

        dynamics[0] = x_dot
        dynamics[1] = y_dot

        return dynamics

    def integrand_cost(self, states, controls, f_prm):
        """ Computation of the integrand cost (Legendre term) """
        # Theta angle
        theta = controls[0]

        x_dot = self.V * np.cos(theta)
        y_dot = self.V * np.sin(theta)

        return x_dot*x_dot + y_dot*y_dot

    def set_initial_guess(self):
        """ Setting of the initial guess for the states, controls, free-parameters
                        and time grid """

        # Time
        self.initial_guess.time = np.linspace(0, 1, self.prm['n_nodes'])

        # States
        self.initial_guess.states = np.ndarray(
            shape=(self.prm['n_states'], self.prm['n_nodes']))

        self.initial_guess.states[0] = np.array(
            [0 for _ in range(len(self.initial_guess.time))])
        self.initial_guess.states[1] = np.array(
            [0 for _ in range(len(self.initial_guess.time))])

        # Controls
        self.initial_guess.controls = np.ndarray(
            shape=(self.prm['n_controls'], self.prm['n_nodes']))

        self.initial_guess.controls[0] = np.array(
            [0 for _ in range(len(self.initial_guess.time))])


if __name__ == '__main__':

    # Instantiation of the problem
    problem = ObstacleAvoidance()

    # Instantiation of the optimization
    optimization = Optimization(problem=problem)

    # Launch of the optimization
    optimization.run()
