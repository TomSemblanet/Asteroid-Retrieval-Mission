#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  01 10:40:21 2021

@author: SEMBLANET Tom

"""

import numpy
import cppad_py

from src.optimal_control.problem import Problem
from src.optimal_control.optimization import Optimization

class User(Problem):
	""" Pattern of an optimal control problem definition """

	def __init__(self):
		""" Initialization of the `User` class """
		n_states = 0
        n_controls = 0
        n_path_con = 0
        n_event_con = 0
        n_f_par = 0
        n_nodes = 0

        Problem.__init__(self, n_states, n_controls, n_path_con,
                         n_event_con, n_f_par, n_nodes)

    def set_constants(self):
        """ Setting of the problem constants """
        pass

    def set_boundaries(self):
        """ Setting of the states, controls, free-parameters, initial and final times
                        boundaries """
       	pass

    def event_constraints(self, xi, ui, xf, uf, ti, tf, f_prm):
        """ Computation of the events constraints """
        events = numpy.ndarray((self.prm['n_event_con'], 1),
                               dtype=cppad_py.a_double)

        return events

    def set_events_constraints_boundaries(self):
        """ Setting of the events constraints boundaries """
        pass

    def path_constraints(self, states, controls, f_par):
        """ Computation of the path constraints """
        paths = np.ndarray((self.prm['n_path_con'],
                            self.prm['n_nodes']), dtype=cppad_py.a_double)

        return paths

    def set_path_constraints_boundaries(self):
        """ Setting of the path constraints boundaries """
        pass

    def dynamics(self, states, controls, f_prm, expl_int=False):
        """ Computation of the states derivatives """
        if expl_int == False:
            dynamics = np.ndarray(
                (states.shape[0], states.shape[1]), dtype=cppad_py.a_double)
        else:
            dynamics = np.zeros(len(states))

        return dynamics

    def end_point_cost(self, ti, xi, tf, xf, f_prm):
        """ Computation of the end point cost (Mayer term) """
        pass

    def integrand_cost(self, states, controls, f_prm):
        """ Computation of the integrand cost (Legendre term) """
        pass

    def set_initial_guess(self):
        """ Setting of the initial guess for the states, controls, free-parameters
                        and time grid """
        pass

if __name__ == '__main__':

    options = {}

    # Instantiation of the problem
    problem = User()

    # Instantiation of the optimization
    optimization = Optimization(problem=problem, **options)

    # Launch of the optimization
    optimization.run()