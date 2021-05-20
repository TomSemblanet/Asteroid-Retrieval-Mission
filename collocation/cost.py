#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  12 11:05:12 2020

@author: SEMBLANET Tom

"""

import cppad_py
import numpy as np

from collocation import utils


class Cost:
    """ `Cost` class manages the costs of the problem namely the Mayer and Legendre parts
        and computes the cost gradient.

        Parameters
        ----------
        problem : Problem
            Optimal-control problem defined by the user
        options : dict
            Transcription and Optimization options dictionnary
        tr_method : Collocation or Pseudospectral
            Transcription method object used to compute the defects constraints

        Attributes
        ----------
        problem : Problem
            Optimal-control problem defined by the user
        mapping_function : <cppad_py dfun object>
            Function mapping the independant and the dependant variables during computation
            of the cost, used by automatic differentiation for the construction of the cost gradient

    """

    def __init__(self, problem, options, tr_method):
        """ Initialization of the `Cost` class """

        # User-defined optimal-control problem
        self.problem = problem

        # General options
        self.options = options

        # Transcription method
        self.tr_method = tr_method

    def compute_cost(self, decision_variables_vector):
        """ Computation of the cost as the sum of both the Mayer and Legendre terms.


        Parameters
        ----------
        decision_variables_vector : array
            Vector of decision variables

        Returns
        -------
        cost_sum : float
            Cost value

        """

        # Initialization of the cost value
        cost_sum = 0

        # Extraction of the states & controls matrices, free parameters, initial et final times
        t_i, t_f, f_prm, states, controls, controls_mid = utils.unpack_decision_variable_vector(
            decision_variables_vector, self.problem.prm)

        # Update of the initial and final times as well as time scaling factor
        self.problem.prm['t_i'] = t_i
        self.problem.prm['t_f'] = t_f
        self.problem.prm['sc_factor'] = (t_f - t_i)/2

        # Computation of the end point cost value (ie. Mayer term)
        if hasattr(self.problem, 'end_point_cost'):
            x_i = states[:, 0]
            x_f = states[:, -1]
            cost_sum += self.problem.end_point_cost(t_i, x_i, t_f, x_f, f_prm)

        # Computation of the integrand cost value (ie. Legendre term)
        if hasattr(self.problem, 'integrand_cost'):
            f_val = self.problem.integrand_cost(states, controls, f_prm)

            if self.options['tr_method'] == 'hermite-simpson':
                # Computation of states at mid-points
                states_mid = self.tr_method.compute_states_col(
                    states, controls, f_prm, self.problem.dynamics, self.problem.prm['sc_factor'])

                # Computation of integrand cost at mid-points
                f_val_mid = self.problem.integrand_cost(
                    states_mid, controls_mid, f_prm)

                cost_sum += self.problem.prm['sc_factor'] * \
                    self.tr_method.quadrature(
                        f_val, f_val_mid)

            else:
                cost_sum += self.problem.prm['sc_factor'] * \
                    self.tr_method.quadrature(f_val)

        return cost_sum

    def set_up_ad(self):
        """ Computation of the mapping function for the cost Gradient """

        # Computation of the mapping function
        self.mapping_function = self.compute_mapping_function()

    def compute_mapping_function(self):
        """ Computation of the mapping function between the independant
            and dependants variables.


        Returns
        -------
        mapping_function : <cppad_py dfun object>
            Mapping function between dependant and independant variables. """

        # False decision variables vector used to launch the
        # `compute_cost` method and record the operations sequence
        ind = np.zeros(self.problem.prm['n_var'], dtype=float)
        ind_ = cppad_py.independent(ind)

        # Cost value returned by the `compute_cost` method
        # Has to create an array as d_fun function takes
        # two arrays in input
        dep_ = np.array([self.compute_cost(ind_)])

        # Construction of the mapping function used latter
        # for the computation of the Gradient vector
        mapping_func = cppad_py.d_fun(ind_, dep_)

        return mapping_func

    def compute_cost_gradient(self, decision_variables_vector):
        """ Computation of the Gradient of the cost function for a
            given decision variables vector.


        Parameters
        ----------
        decision_variables_vector : array
            Vector of decision variables

        Returns
        -------
        grad : array
            Gradient of the cost function wrt. each variables.

        """

        # Computation of the Gradient matrix
        grad = self.mapping_function.jacobian(decision_variables_vector)[0]

        return grad
