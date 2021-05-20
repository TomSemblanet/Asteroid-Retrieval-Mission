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


class MoonLander(Problem):
    """ Moon Lander : optimal control problem """

    def __init__(self):
        """ Initialization of the `MoonLander` class """
        n_states = 3
        n_controls = 1
        n_path_con = 0
        n_event_con = 5
        n_f_par = 0
        n_nodes = 300

        Problem.__init__(self, n_states, n_controls, n_path_con,
                         n_event_con, n_f_par, n_nodes)

    def set_constants(self):
        """ Setting of the problem constants """
        self.g = 1
        self.E = 2.349

    def set_boundaries(self):
        """ Setting of the states, controls, free-parameters, initial and final times
                        boundaries """
        # States boundaries
        self.low_bnd.states[0] = -20
        self.upp_bnd.states[0] = 20

        self.low_bnd.states[1] = -20
        self.upp_bnd.states[1] = 20

        self.low_bnd.states[2] = 0.01
        self.upp_bnd.states[2] = 1

        # Controls boundaries
        self.low_bnd.controls[0] = 0
        self.upp_bnd.controls[0] = 1.227

        # Initial and final times boundaries
        self.low_bnd.ti = self.upp_bnd.ti = 0
        self.low_bnd.tf = 0
        self.upp_bnd.tf = 2

    def event_constraints(self, xi, ui, xf, uf, ti, tf, f_prm):
        """ Computation of the events constraints """
        events = np.ndarray((self.prm['n_event_con'], 1),
                            dtype=cppad_py.a_double)

        h_i = xi[0]
        v_i = xi[1]
        m_i = xi[2]

        h_f = xf[0]
        v_f = xf[1]

        events[0] = h_i
        events[1] = v_i
        events[2] = m_i
        events[3] = h_f
        events[4] = v_f

        return events

    def set_events_constraints_boundaries(self):
        """ Setting of the events constraints boundaries """
        self.low_bnd.event[0] = self.upp_bnd.event[0] = 1

        self.low_bnd.event[1] = self.upp_bnd.event[1] = - 0.783

        self.low_bnd.event[2] = self.upp_bnd.event[2] = 1

        self.low_bnd.event[3] = self.upp_bnd.event[3] = 0

        self.low_bnd.event[4] = self.upp_bnd.event[4] = 0

    def dynamics(self, states, controls, f_prm, expl_int=False):
        """ Computation of the states derivatives """
        if expl_int == False:
            dynamics = np.ndarray(
                (states.shape[0], states.shape[1]), dtype=cppad_py.a_double)
        else:
            dynamics = np.zeros(len(states))

        h = states[0]
        v = states[1]
        m = states[2]

        T = controls[0]

        h_dot = v
        v_dot = T/m - self.g
        m_dot = -T/self.E

        dynamics[0] = h_dot
        dynamics[1] = v_dot
        dynamics[2] = m_dot

        return dynamics

    def integrand_cost(self, states, controls, f_prm):
        """ Computation of the integrand cost (Legendre term) """
        T = controls[0]
        return np.array([T_ for T_ in T])

    def set_initial_guess(self):
        """ Setting of the initial guess for the states, controls, free-parameters
                        and time grid """

        # Time
        self.initial_guess.time = np.linspace(0, 1.4, self.prm['n_nodes'])

        # States
        self.initial_guess.states[0] = np.array(
            [.5 for _ in range(self.prm['n_nodes'])])
        self.initial_guess.states[1] = np.array(
            [.5 for _ in range(self.prm['n_nodes'])])
        self.initial_guess.states[2] = np.array(
            [.75 for _ in range(self.prm['n_nodes'])])

        # Controls
        self.initial_guess.controls[0] = np.array(
            [.6 for _ in range(len(self.initial_guess.time))])


if __name__ == '__main__':

    # Instantiation of the problem
    problem = MoonLander()

    # Instantiation of the optimization
    optimization = Optimization(problem=problem)

    # Run of the optimization
    optimization.run()
