#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  14 09:50:23 2020

@author: SEMBLANET Tom

"""
 
import cppad_py
import numpy

from src.optimal_control.problem import Problem
from src.optimal_control.optimization import Optimization


class VanDerPolOscillator(Problem):
    """ Van der pol oscillator optimal control problem """

    def __init__(self):
        """ Initialization of the `VanDerPolOscillator` class """
        n_states = 2
        n_controls = 1
        n_path_con = 0
        n_event_con = 3
        n_f_par = 0
        n_nodes = 100

        Problem.__init__(self, n_states, n_controls, n_path_con,
                         n_event_con, n_f_par, n_nodes)

    def set_constants(self):
        """ Setting of the problem constants """
        self.r = 0.2

    def set_boundaries(self):
        """ Setting of the states, controls, free-parameters, initial and final times
                        boundaries """
        # States boundaries
        self.low_bnd.states[0] = -2
        self.upp_bnd.states[0] = 2

        self.low_bnd.states[1] = -0.4
        self.upp_bnd.states[1] = 1

        # Controls boundaries
        self.low_bnd.controls[0] = -1
        self.upp_bnd.controls[0] = 1

        # Initial and final times boundaries
        self.low_bnd.ti = self.upp_bnd.ti = 0
        self.low_bnd.tf = self.upp_bnd.tf = 4

    def event_constraints(self, xi, ui, xf, uf, ti, tf, f_prm):
        """ Computation of the events constraints """
        events = numpy.ndarray((self.prm['n_event_con'], 1),
                               dtype=cppad_py.a_double)

        # Initial values of states are constrained
        x1i = xi[0]
        x2i = xi[1]

        # Final values of state n°1 is constrained
        x1f = xf[0]

        events[0] = x1i
        events[1] = x2i
        events[2] = x1f

        return events

    def set_events_constraints_boundaries(self):
        """ Setting of the events constraints boundaries """
        self.low_bnd.event[0] = self.upp_bnd.event[0] = 1

        self.low_bnd.event[1] = self.upp_bnd.event[1] = 1

        self.low_bnd.event[2] = self.upp_bnd.event[2] = self.r*self.r

    def dynamics(self, states, controls, f_prm, expl_int=False):
        """ Computation of the states derivatives """
        if expl_int == False:
            dynamics = numpy.ndarray(
                (states.shape[0], states.shape[1]), dtype=cppad_py.a_double)
        else:
            dynamics = numpy.zeros(len(states))

        x1 = states[0]
        x2 = states[1]

        u = controls[0]

        x1_dot = x2
        x2_dot = -x1 + x2*(1 - x1*x1) + u

        dynamics[0] = x1_dot
        dynamics[1] = x2_dot

        return dynamics

    def integrand_cost(self, states, controls, f_prm):
        """ Computation of the integrand cost (Legendre term) """
        x1 = states[0]
        x2 = states[1]
        u = controls[0]

        return [x1_*x1_ + x2_*x2_ + u_*u_ for x1_, x2_, u_ in zip(x1, x2, u)]

    def set_initial_guess(self):
        """ Setting of the initial guess for the states, controls, free-parameters
                        and time grid """
        # Time grid initial guess
        self.initial_guess.time = numpy.linspace(0, 4, self.prm['n_nodes'])

        # States initial guess
        self.initial_guess.states[0] = numpy.array(
            [0 for _ in self.initial_guess.time])
        self.initial_guess.states[1] = numpy.array(
            [0 for _ in self.initial_guess.time])

        # Controls initial guess
        self.initial_guess.controls[0] = numpy.array(
            [0 for _ in self.initial_guess.time])


if __name__ == '__main__':

    # Instantiation of the problem
    problem = VanDerPolOscillator()

    # Instantiation of the optimization
    optimization = Optimization(problem=problem)

    # Launch of the optimization
    optimization.run()
