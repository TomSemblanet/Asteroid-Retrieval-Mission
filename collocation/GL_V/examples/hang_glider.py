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


class HangGlider(Problem):
    """ Hang Glider : optimal control problem """

    def __init__(self):
        """ Initialization of the `HangGlider` class """
        n_states = 4
        n_controls = 1
        n_st_path_con = 0
        n_ct_path_con = 0
        n_event_con = 7
        n_f_par = 0
        n_nodes = 200

        Problem.__init__(self, n_states, n_controls, n_st_path_con, n_ct_path_con,
                         n_event_con, n_f_par, n_nodes)

    def set_constants(self):
        """ Setting of the problem constants """
        self.u_M = 2.5
        self.R = 100.0
        self.C0 = .034
        self.k = .069662
        self.m = 100
        self.S = 14
        self.rho = 1.13
        self.g = 9.80665

    def set_boundaries(self):
        """ Setting of the states, controls, free-parameters, initial and final times
                        boundaries """
        # States boundaries
        self.low_bnd.states[0] = 0
        self.upp_bnd.states[0] = 1500

        self.low_bnd.states[1] = 0
        self.upp_bnd.states[1] = 1100

        self.low_bnd.states[2] = 0
        self.upp_bnd.states[2] = 15

        self.low_bnd.states[3] = -4
        self.upp_bnd.states[3] = 4

        # Controls boundaries
        self.low_bnd.controls[0] = 0
        self.upp_bnd.controls[0] = 1.4

        # Initial and final times boundaries
        self.low_bnd.ti = self.upp_bnd.ti = 0
        self.low_bnd.tf = 0.1
        self.upp_bnd.tf = 200

    def event_constraints(self, xi, ui, xf, uf, ti, tf, f_prm):
        """ Computation of the events constraints """
        events = np.ndarray((self.prm['n_event_con'], 1),
                            dtype=cppad_py.a_double)

        x_i = xi[0]
        y_i = xi[1]
        vx_i = xi[2]
        vy_i = xi[3]

        y_f = xf[1]
        vx_f = xf[2]
        vy_f = xf[3]

        events[0] = x_i
        events[1] = y_i
        events[2] = vx_i
        events[3] = vy_i
        events[4] = y_f
        events[5] = vx_f
        events[6] = vy_f

        return events

    def set_events_constraints_boundaries(self):
        """ Setting of the events constraints boundaries """
        self.low_bnd.event[0] = self.upp_bnd.event[0] = 0

        self.low_bnd.event[1] = self.upp_bnd.event[1] = 1e3

        self.low_bnd.event[2] = self.upp_bnd.event[2] = 13.2275675

        self.low_bnd.event[3] = self.upp_bnd.event[3] = -1.28750052

        self.low_bnd.event[4] = self.upp_bnd.event[4] = 900

        self.low_bnd.event[5] = self.upp_bnd.event[5] = 13.2275675

        self.low_bnd.event[6] = self.upp_bnd.event[6] = -1.28750052

    def dynamics(self, states, controls, f_prm, expl_int=False):
        """ Computation of the states derivatives """
        if expl_int == False:
            dynamics = np.ndarray(
                (states.shape[0], states.shape[1]), dtype=cppad_py.a_double)
        else:
            dynamics = np.zeros(len(states))

        x = states[0]
        y = states[1]
        vx = states[2]
        vy = states[3]

        C_L = controls[0]

        C_D = self.C0 + self.k*C_L*C_L
        vr = (vx*vx + vy*vy)**.5
        D = .5 * C_D * self.rho * self.S * vr * vr
        L = .5 * C_L * self.rho * self.S * vr * vr
        X = (x/self.R - 2.5)*(x/self.R - 2.5)
        ua = self.u_M * (1-X) * np.exp(-X)
        Vy = vy - ua
        sin_eta = Vy / vr
        cos_eta = vx / vr
        W = self.m*self.g

        x_dot = vx
        y_dot = vy
        vx_dot = 1/self.m * (-L*sin_eta - D*cos_eta)
        vy_dot = 1/self.m * (L*cos_eta - D*sin_eta - W)

        dynamics[0] = x_dot
        dynamics[1] = y_dot
        dynamics[2] = vx_dot
        dynamics[3] = vy_dot

        return dynamics

    def end_point_cost(self, ti, xi, tf, xf, f_prm):
        """ Computation of the end point cost (Mayer term) """
        xf = xf[0]
        return - xf

    def set_initial_guess(self):
        """ Setting of the initial guess for the states, controls, free-parameters
                        and time grid """

        # Time
        self.initial_guess.time = np.linspace(0, 105, self.prm['n_nodes'])

        # States
        self.initial_guess.states = np.ndarray(
            shape=(self.prm['n_states'], self.prm['n_nodes']))

        self.initial_guess.states[0] = np.linspace(
            0, 1000, self.prm['n_nodes'])
        self.initial_guess.states[1] = np.linspace(
            1e3, 9e2, self.prm['n_nodes'])
        self.initial_guess.states[2] = np.array(
            [13.23 for _ in range(self.prm['n_nodes'])])
        self.initial_guess.states[3] = np.array(
            [-1.288 for _ in range(self.prm['n_nodes'])])

        # Controls
        self.initial_guess.controls = np.ndarray(
            shape=(self.prm['n_controls'], self.prm['n_nodes']))

        self.initial_guess.controls[0] = np.array(
            [1 for _ in range(self.prm['n_nodes'])])


if __name__ == '__main__':

    # Instantiation of the problem
    problem = HangGlider()

    # Instantiation of the optimization
    optimization = Optimization(problem=problem)

    # Launch of the optimization
    optimization.run()
